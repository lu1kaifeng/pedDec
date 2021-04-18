const { createProxyMiddleware } = require('http-proxy-middleware');
var cors = require('cors')
module.exports = function(app) {
  app.use(
    ['/socket.io','/sockjs-node','/api'],
    createProxyMiddleware({
      target: 'http://localhost:5000',
      changeOrigin: true,
      ws:true
    })
  );

  app.options('*', cors())
};
