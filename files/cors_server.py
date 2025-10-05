#this code is for hosting the images with allowing all origins to access without permissions
# todo change the code so that it gets the url dynamically from the folder
# cors_server.py
from http.server import SimpleHTTPRequestHandler, HTTPServer

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')  # allow all origins
        super().end_headers()

PORT = 9900
server = HTTPServer(('0.0.0.0', PORT), CORSRequestHandler)
print(f"Serving on http://localhost:{PORT}")
server.serve_forever()
