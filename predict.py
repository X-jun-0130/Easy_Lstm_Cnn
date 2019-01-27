import numpy as np
from Lstm_Cnn import Lstm_CNN
import tensorflow as tf
from data_processing import read_category, get_wordid, get_word2vec, process, batch_iter, seq_length
from Parameters import Parameters as pm

def val():

    pre_label = []
    label = []
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint('./checkpoints/Lstm_CNN')
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    val_x, val_y = process(pm.val_filename, wordid, cat_to_id, max_length=pm.seq_length)
    batch_val = batch_iter(val_x, val_y, batch_size=64)
    for x_batch, y_batch in batch_val:
        real_seq_len = seq_length(x_batch)
        feed_dict = model.feed_data(x_batch, y_batch, real_seq_len, 1.0)
        pre_lab = session.run(model.predict, feed_dict=feed_dict)
        pre_label.extend(pre_lab)
        label.extend(y_batch)
    return pre_label, label


if __name__ == '__main__':

    pm = pm
    sentences = []
    label2 = []
    categories, cat_to_id = read_category()
    wordid = get_wordid(pm.vocab_filename)
    pm.vocab_size = len(wordid)
    pm.pre_trianing = get_word2vec(pm.vector_word_npz)

    model = Lstm_CNN()
    pre_label, label = val()
    correct = np.equal(pre_label, np.argmax(label, 1))
    accuracy = np.mean(np.cast['float32'](correct))
    print('accuracy:', accuracy)
    print("预测前10项：", ' '.join(str(pre_label[:10])))
    print("正确前10项：", ' '.join(str(np.argmax(label[:10], 1))))

