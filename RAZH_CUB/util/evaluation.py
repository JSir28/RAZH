import numpy as np

def presicion_and_recall(query_h, database_h, top_nums,query_label, database_label):
    St = np.dot(query_label, np.transpose(database_label))  #计算查询图片标签和数据库标签的关系  ij为1即查询i和库j是同类
    Wt = np.float32(St>0)  #将St转为flaot

    query_num = query_h.shape[0]  #查询数据数量
    database_num = database_h.shape[0] #查询库数据量
    nbits = query_h.shape[1] #哈希码位数
    D2 = np.dot(query_h, np.transpose(database_h)) #计算查询图片哈系码和数据库哈系码的关系 数值越大越相似
    ham_dist = (nbits - D2) / 2. #哈系距离

    sort_indices = np.argsort(ham_dist, axis=1)  #将查询库按照哈希距离排序
    maximum_indices = np.argsort(-St, axis=1) #将与query具有相同标签的图像找到
    query_sorted_Wt = np.zeros(shape=Wt.shape, dtype=np.float32)
    query_sorted_St = np.zeros(shape=St.shape, dtype=np.float32)
    maximum_sorted_St = np.zeros(shape=St.shape, dtype=np.float32)

    for i in range(query_num):
        query_sorted_Wt[i, :] = Wt[i, sort_indices[i, :]] #哈希码相近的标签情况
        maximum_sorted_St[i, :] = St[i, maximum_indices[i, :]] #标签相似的
        query_sorted_St[i, :] = St[i, sort_indices[i, :]]

    # ap = np.zeros(shape=(query_num, len(top_nums)), dtype=np.float32)
    # ap2 = np.zeros(shape=(query_num, len(top_nums)), dtype=np.float32)
    map = np.zeros(shape=(len(top_nums)), dtype=np.float32)
    ndcg = np.zeros(shape=(len(top_nums)), dtype=np.float32)
    acg = np.zeros(shape=(len(top_nums)), dtype=np.float32)
    presicion = np.zeros(shape=(len(top_nums)), dtype=np.float32)
    recall = np.zeros(shape=(len(top_nums)), dtype=np.float32)
    wap = np.zeros(shape=(len(top_nums)), dtype=np.float32)

    cum_Wt = np.cumsum(query_sorted_Wt, axis=1)
    cum_St = np.cumsum(query_sorted_St, axis=1)

    c = np.tile(np.reshape(range(1, database_num + 1), (1, database_num)), (query_num, 1))
    cum_Wt_div = np.true_divide(cum_Wt, c)
    cum_St_div = np.true_divide(cum_St, c)

    dcg = np.true_divide((2 ** query_sorted_St) - 1, np.log(c + 1))
    maximum_dcg = np.true_divide((2 ** maximum_sorted_St) - 1, np.log(c + 1))

    for ii in range(len(top_nums)):
        topk = top_nums[ii]
        tmp_cum_Wt_div = cum_Wt_div[:, 0:topk]
        tmp_cum_St_div = cum_St_div[:, 0:topk]
        tmp_sorted_Wt = query_sorted_Wt[:, 0:topk]
        tmp_dcg = dcg[:, 0:topk]
        tmp_maximum_dcg = maximum_dcg[:, 0:topk]

        map[ii] = np.mean(np.nan_to_num(
            np.true_divide(np.sum(np.multiply(tmp_cum_Wt_div, tmp_sorted_Wt), axis=1), np.sum(tmp_sorted_Wt, axis=1))))
        wap[ii] = np.mean(np.nan_to_num(
            np.true_divide(np.sum(np.multiply(tmp_cum_St_div, tmp_sorted_Wt), axis=1), np.sum(tmp_sorted_Wt, axis=1))))

        _dcg = np.true_divide(np.sum(tmp_dcg, axis=1), np.sum(tmp_maximum_dcg, axis=1))
        # _dcg[np.isnan(_dcg)] = 0
        ndcg[ii] = np.mean(_dcg)
        acg[ii] = np.mean(query_sorted_St[:, 0:topk])
        # map[ii] = np.mean(ap[:, ii])
        # wap[ii] = np.mean(ap2[:, ii])
        presicion[ii] = np.mean(tmp_sorted_Wt)
        recall[ii] = np.true_divide(np.sum(tmp_sorted_Wt), np.sum(query_sorted_Wt))
    return presicion, recall, map, wap, acg
