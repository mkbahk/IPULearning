//--------------------hello-server-p7777.js-----------------//
var os = require('os');
var http = require('http');
var handleRequest = function(request, response) {
  response.writeHead(200);
  response.end("What's up, man! I'm "+os.hostname()+", running for Graphocore IPUs.");
  //log
  console.log("["+Date(Date.now()).toLocaleString()+"] "+os.hostname());
}
var www = http.createServer(handleRequest);
www.listen(7777);
//--------------------hello-server-p7777.js-----------------//


%node hello-server-p7777.js

//----------
Dockerfile 
//----------
FROM node:carbon
EXPOSE 7777
COPY server.js .
CMD node server.js > log.out
//----------

% docker build -t mkbahk/hello-node:v1 .

%docker run -d -p 7777:7777 mkbahk/hello-node:v1

% docker exec -i -t [컨테이너 ID] /bin/bash
