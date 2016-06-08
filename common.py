# coding=utf-8

# import matplotlib as mpl
# mpl.rcParams['font.sans-serif'] = ['STHeiti']
# mpl.use('TKAgg')
# from matplotlib import pyplot
# import re
# from full2half import full2half
import connector
import datetime

user = 'bigdata'
now = datetime.datetime.now()


# connect mysql database
def db_connect(sql, values=None):
    conn = connector.Connect()
    db = conn.db
    cursor = db.cursor()
    # sql = 'select id, content from t_po_dianping'
    num = cursor.execute(sql, values)
    result = cursor.fetchmany(num)
    cursor.close()
    db.commit()
    db.close()

    return result


def get_comments(type):
    if type == 'train':
        sql = 'select emotion, commtxt from t_dianping_train_01 where emotion is not null order by id'
    elif type == 'predict':
        sql = 'select id, commtxt from t_dianping_predict_02 order by id'
    else:
        print 'Error: Wrong type input,please try again!'

    return db_connect(sql)


def update_words(list_sorted):
    global user, now
    sql = 'truncate table t_po_words'
    db_connect(sql)
    for row in list_sorted:
        sql = 'insert into t_po_words(word, num, create_user, create_date, modify_user, modify_date, del_flag)' \
              'VALUES (%s, %s, %s, %s, %s, %s, %s)'
        values = (row[0], row[1], user, now, user, now, 0)
        db_connect(sql, values)

    return len(list_sorted)


def get_words():
    sql = 'select word, id from t_po_words order by id'
    result = db_connect(sql)

    return result


def update_predict(p_label, idx):
    global user, now
    # sql = 'update t_dianping_predict set predict=%s, modify_user=%s, modify_date=%s, labeled_by=%s where id=%s'
    sql = 'update t_dianping_predict_02 set predict=%s where id=%s'
    for i in range(len(p_label)):
        db_connect(sql, (p_label[i], idx[i]))


# def plot_topwords(self, dict):
#     print "正在绘制柱状图..."
#     dict_sorted = sorted(dict.iteritems(), key=lambda d: d[1], reverse=True)
#     bar_width = 0.35
#     pyplot.bar(range(30), [dict_sorted[i][1] for i in range(30)], bar_width)
#     pyplot.xticks(range(30), [dict_sorted[i][0] for i in range(30)], rotation=30)
#     pyplot.title(u"评论" + u"by Maxiee")
#     pyplot.show()


if __name__ == '__main__':
    pass
