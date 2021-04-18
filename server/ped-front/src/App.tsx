import React, { useEffect, useState } from 'react'
import { BrowserRouter, Switch, Route } from 'react-router-dom'

import { io } from 'socket.io-client'
import axios from 'axios'
import { Navbar } from './components/Navbar'
import { About } from './pages/About'
import { Home } from './pages/Home'

function useFrame() {
  const [frame, setFrame] = useState<Blob | null>(null)
  useEffect(() => {
    const socket = io()
    socket.on('frame', (args: ArrayBuffer) => {
      setFrame(new Blob([args], { type: 'image/jpeg' }))
    })
    axios.get('/api')
  }, [])
  return frame
}

function useTrack() {
  const [track, setTrack] = useState<any | null>(null)
  useEffect(() => {
    const socket = io()
    socket.on('track', (args) => {
      setTrack(args)
    })
    axios.get('/api')
  }, [])
  return track
}

const App: React.FC = () => {
  const frame = useFrame()
  useTrack()
  return (
    <BrowserRouter>
      <Navbar />
      <div className="container">
        <Switch>
          {frame !== null && <img width={1024} src={window.URL.createObjectURL(frame)} alt='penis'/>}
          <Route path="/" component={Home} exact />
          <Route path="/about" component={About} />
        </Switch>
      </div>
    </BrowserRouter>
  )
}

export default App
