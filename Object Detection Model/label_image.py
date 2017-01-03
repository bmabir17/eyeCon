import tensorflow as tf
import os, sys
folder="/tf_files/input/"
for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
        print("using",filename)
        # change this as you see fit
        image_path = folder+filename
        #print("path: ",image_path)
        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        # Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line 
                           in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

        # Unpersists graph from file
        with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            
            predictions = sess.run(softmax_tensor, \
                     {'DecodeJpeg/contents:0': image_data})
            
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            # print(top_k)
            # for node_id in top_k:
            node_id=top_k[0]
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
    else:
        print("No jpg file is found")
