import { useEffect, useState } from 'react'
import { io } from 'socket.io-client'
import axios from 'axios'

export interface Pedestrian {
  id: number
  image: ArrayBuffer
  age: number
  confidence: number
}

export function useFrame() {
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

export function useTrack() : Pedestrian[] | null{
  const [track, setTrack] = useState<Pedestrian[] | null>(null)
  useEffect(() => {
    const socket = io()
    socket.on('track', (args: Pedestrian[]) => {
      setTrack(args)
    })
    axios.get('/api')
  }, [])
  return track
}
