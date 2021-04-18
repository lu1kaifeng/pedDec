import React, { useEffect } from 'react'
import { BrowserRouter, Switch, Route } from 'react-router-dom'

import { io } from 'socket.io-client'
import { Navbar } from './components/Navbar'
import { About } from './pages/About'
import { Home } from './pages/Home'

function useFrame() {
  // const [isOnline, setIsOnline] = useState(null)
  useEffect(() => {
    const socket = io()
    socket.on('frame', (args) => {
      console.log(args)
    })
  })
}

const App: React.FC = () => {
  useFrame()
  return (
    <BrowserRouter>
      <Navbar />
      <div className="container">
        <Switch>
          <Route path="/" component={Home} exact />
          <Route path="/about" component={About} />
        </Switch>
      </div>
    </BrowserRouter>
  )
}

export default App
