from aiohttp import web
import socketio

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

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
    
    await sio.emit('message', data, room=sid)

@sio.on('disconnect')
def disconnect(sid):
    print('disconnect ', sid)

app.router.add_get('/', index)

if __name__ == '__main__':
    web.run_app(app)