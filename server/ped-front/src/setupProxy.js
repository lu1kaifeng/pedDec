const { createProxyMiddleware } = require('http-proxy-middleware');
var cors = require('cors')
module.exports = function(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://localhost:8000',
      changeOrigin: true,
    })
  );

  app.options('*', cors())
};
