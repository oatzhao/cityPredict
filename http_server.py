#coding = utf-8

from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from socketserver import ThreadingMixIn

hostIP = ''
portNum = 8080
class mySoapServer(BaseHTTPRequestHandler):
    def do_head(self):
        pass

    def do_GET(self):
        try:
            self.send_response(200, message=None)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            res = '''
            <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">

           <HTML>

           <HEAD><META content="IE=5.0000" http-equiv="X-UA-Compatible">

           <META content="text/html; charset=gb2312" http-equiv=Content-Type>

           </HEAD>

           <BODY>

           Hi, www.perlcn.com is a good site to learn python!

           </BODY>

           </HTML>

           '''
            self.wfile.write(res.encode(encoding='utf_8',errors='strict'))
        except IOError:
            self.send_error(404, message=None)

    def do_POST(self):
        try:
            self.send_response(200, message=None)
            self.send_header('Content-type', 'text/html')
            self.send_header()
            res='''
             <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">

             <HTML>

             <HEAD><META content="IE=5.0000" http-equiv="X-UA-Compatible">

             <META content="text/html; charset=gb2312" http-equiv=Content-Type>

             </HEAD>

             <BODY>

             Hi, www.perlcn.com is a good site to learn python!

             </BODY>

             </HTML>

             '''
            self.wfile.write(res.encode(encoding='utf_8', errors='strict'))
        except IOError:
            self.send_error(404, message=None)

class ThreadingHttpServer(ThreadingMixIn, HTTPServer):
    pass

# myServer = ThreadingHttpServer((hostIP, portNum), mySoapServer)
#
# myServer.serve_forever()
# myServer.server_close()

# a = tf.constant([[1.,2.,3.],[4.,5.,6.]])
# b = np.float32(np.random.randn(3, 2))
# c = tf.matmul(a, b)
# init = tf.global_variables_initializer()
# sess = tf.Session()
# with tf.Session() as sess:
#     print(c.eval())
# sess = tf.InteractiveSession()
# print(c.eval())