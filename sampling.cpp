int randint_(int end)
{
    return rand() % end;
}

py::array_t<int> sample_negative(int user_num, int item_num, int train_num, std::vector<std::vector<int>> allPos, int neg_num)
{
    int perUserNum = (train_num / user_num);
    int row = neg_num + 2;
    py::array_t<int> S_array = py::array_t<int>({user_num * perUserNum, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user = 0; user < user_num; user++)
    {
        std::vector<int> pos_item = allPos[user];
        for (int user_id = 0; user_id < user_num; user_id++) {
        positive_list = allPos[user_id];
        for (int i = 0; i < positive_list.size(); i++) {
            for (int t = 0; t < neg_num; t++) {
                int item_j = rand() % item_num;
                while (item_j in positive_list) {
                    item_j = rand() % item_num;
                }
                features_fill.append({item_j});
            }
        }
    }
    S_array = features_fill;
    return S_array;
}