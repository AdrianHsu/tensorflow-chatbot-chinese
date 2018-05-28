# ckpt-00000 should be modified
python3 -m tensorflow.python.tools.freeze_graph --input_graph=load/test.pb --input_checkpoint=save/chatbot.ckpt-212000 --output_graph=load/frozen.pb --output_node_names=decoder/decoder_pred_eval --input_binary=true
