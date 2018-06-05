from aiohttp import web
import socketio
import inference 
import pickle
import tensorflow as tf


sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)
x1 = None
x2 = None
x3 = None
y = None
graph = None
sess = None

word2idx = {}
idx2word = {}

async def index(request):
    """Serve the client-side application."""
    with open('index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

@sio.on('message')
async def message(sid, data):
    print(data, ", sid: ", sid)
    test_input = data
    input_id = inference.seq(test_input, word2idx)
    y_out = sess.run(y, feed_dict={
        x1: [input_id],
        x2: [len(input_id)],
        x3: 1
    })

    sen = inference.dec(y_out[0], idx2word)
    print("chatbot: " + sen)

    await sio.emit('message', sen, room=sid)

@sio.on('disconnect')
def disconnect(sid):
    print('disconnect ', sid)

app.router.add_get('/', index)

if __name__ == '__main__':

    graph = inference.load_graph("frozen.pb")
    sess = tf.InteractiveSession(graph=graph)
    x1 = graph.get_tensor_by_name('prefix/encoder_inputs:0')
    x2 = graph.get_tensor_by_name('prefix/encoder_inputs_length:0')
    x3 = graph.get_tensor_by_name('prefix/batch_size:0')
    y = graph.get_tensor_by_name('prefix/decoder/decoder_pred_eval:0')
    
    with open('word2idx.pkl', 'rb') as handle:
        word2idx = pickle.load(handle)
    with open('idx2word.pkl', 'rb') as handle:
        idx2word = pickle.load(handle)
    input_id = inference.seq('測試', word2idx)
    print('測試: ', input_id)
    print('load pickle: done')


    web.run_app(app)