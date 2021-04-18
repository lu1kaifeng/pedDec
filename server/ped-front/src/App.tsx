import React, { } from 'react'
import { BrowserRouter } from 'react-router-dom'

import { Navbar } from './components/Navbar'

import { useFrame, useTrack } from './Client'
import PedCard from './PedCard'



const App: React.FC = () => {
  const frame = useFrame()
  const track = useTrack()
  return (
    <BrowserRouter>
      <Navbar />
      <div className="container">
          {frame !== null && <img width={1024} src={window.URL.createObjectURL(frame)} alt='penis'/>}
        {/* eslint-disable-next-line react/jsx-props-no-spreading */}
          {track !== null && track.map((value)=><PedCard {...value}/>)}
      </div>
    </BrowserRouter>
  )
}

export default App
