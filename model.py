class FairLoss(torch.nn.Module):
    def __init__(self, fairness_factor):
        super(FairLoss, self).__init__()
        self.fairness_factor = fairness_factor

    def forward(self, pos_scores, neg_scores):
        diff = pos_scores - neg_scores
        fair_loss = -torch.log(torch.sigmoid(self.fairness_factor * diff))
        return fair_loss.mean()
    
class FairGCF(BasicModel):
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def cosine_similarity(self, x, y):
        x = x - torch.mean(x)
        y = y - torch.mean(y)
        xy = torch.matmul(x, y.transpose(0, 1))
        x_norm = torch.sqrt(torch.mul(x, x).sum(1))
        y_norm = torch.sqrt(torch.mul(y, y).sum(1))
        x_norm = 1.0 / (x_norm.unsqueeze(1) + 1e-8)
        y_norm = 1.0 / (y_norm.unsqueeze(0) + 1e-8)
        l = 5
        num_b = x.shape[0] // l
        if num_b * l < x.shape[0]:
            l = l + 1
        for i in range(l):
            begin = i * num_b
            end = (i + 1) * num_b
            end = xy.shape[0] if end > xy.shape[0] else end
            xy[begin:end] = torch.mul(torch.mul(xy[begin:end], x_norm[begin:end]), y_norm)
        return xy

    def tile(self, a, reps):
        expanded_a = a.unsqueeze(1)
        tiled_a = expanded_a.repeat(1, *reps).view(-1)
        return tiled_a

    def top_sim(self, sim_adj, toph, num_node):
        sim_node = torch.topk(sim_adj, k=toph + self.self_loop, dim=1)
        sim_node_value = sim_node.values[:, self.self_loop:].reshape((-1)) / toph
        sim_node_col = sim_node.indices[:, self.self_loop:].reshape((-1))
        sim_node_row = torch.tensor(range(num_node)).long().reshape((-1, 1))
        sim_node_row = self.tile(sim_node_row, [1, toph]).reshape((-1))
        sim_node_indices = torch.stack([sim_node_row.to(world.device), sim_node_col.to(world.device)])
        sim_adj = torch.sparse.FloatTensor(sim_node_indices, sim_node_value, torch.Size(sim_adj.shape))
        return sim_adj
    def get_consin_adj(self, users_emb, items_emb):
        with torch.no_grad():
            user_sim = self.cosine_similarity(users_emb, users_emb)
            self.user_sim_adj = self.top_sim(user_sim, self.top_H, self.num_users)
            del user_sim
            item_sim = self.cosine_similarity(items_emb, items_emb)
            self.item_sim_adj = self.top_sim(item_sim, self.top_H, self.num_items)
            del item_sim
    def get_interaction_adj(self, emb_u, emb_i):
        with torch.no_grad():
            sim = torch.sigmoid(torch.matmul(emb_u, emb_i.transpose(0, 1)))
            adj = self.adj.to_dense()
                adj = (1 - self.beta) * adj * sim + self.beta * adj
        return adj

    
    #FairGCF
    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg, fairness_factor):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        # 计算公平性损失
        fair_criterion = FairLoss(fairness_factor)
        fair_loss = fair_criterion(pos_scores, neg_scores)
        # 计算BPR损失
        bpr_losses = -torch.log(torch.sigmoid(pos_scores - neg_scores))
        # 取平均损失
        bpr_loss = bpr_losses.mean()
        # 将公平性损失与BPR损失相结合
        loss = bpr_loss + fair_loss
        return loss, reg_loss