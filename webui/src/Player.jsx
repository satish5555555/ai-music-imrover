import React, { useRef } from 'react'
import { Midi } from '@tonejs/midi'
import * as Tone from 'tone'

export default function Player({ midiUrl }) {
  const isPlaying = useRef(false)

  async function play() {
    if (isPlaying.current) return
    isPlaying.current = true
    try {
      const res = await fetch(midiUrl)
      const ab = await res.arrayBuffer()
      const midi = new Midi(ab)
      await Tone.start()
      const now = Tone.now()
      const synth = new Tone.PolySynth(Tone.Synth).toDestination()
      midi.tracks.forEach(track => {
        track.notes.forEach(note => {
          synth.triggerAttackRelease(note.name, note.duration, now + note.time, note.velocity)
        })
      })
      const dur = midi.duration || 10
      setTimeout(()=>{ isPlaying.current = false }, (dur+1)*1000)
    } catch (e) {
      console.error("play error", e)
      isPlaying.current = false
    }
  }

  return (
    <div className="player">
      <button onClick={play}>Play</button>
      <a className="btn" href={midiUrl} download>Download</a>
    </div>
  )
}
