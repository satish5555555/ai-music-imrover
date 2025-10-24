import React, { useState, useEffect } from 'react'
import UploadForm from './UploadForm'
import Player from './Player'

export default function App() {
  const [outputs, setOutputs] = useState([])
  const [status, setStatus] = useState('idle')

  async function fetchList() {
    const r = await fetch('/api/list').catch(()=>null)
    if(!r) return
    const j = await r.json()
    setOutputs(j.outputs || [])
  }

  useEffect(()=>{ fetchList() }, [])

  return (
    <div className="container">
      <h1>AI Music Improver</h1>
      <UploadForm onDone={()=>{ setStatus('done'); fetchList(); }} setStatus={setStatus}/>
      <div className="status">Status: {status}</div>

      <h2>Improved outputs</h2>
      <button onClick={fetchList}>Refresh</button>
      <div className="outputs">
        {outputs.length === 0 && <div>No outputs yet</div>}
        {outputs.map((f) => (
          <div key={f} className="output-card">
            <div className="fname">{f}</div>
            <a className="btn" href={`/api/download/${f}`} target="_blank" rel="noreferrer">Download</a>
            <Player midiUrl={`/api/download/${f}`} />
          </div>
        ))}
      </div>
    </div>
  )
}
