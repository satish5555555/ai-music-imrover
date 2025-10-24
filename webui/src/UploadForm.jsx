import React, { useState } from 'react'

export default function UploadForm({ onDone, setStatus }) {
  const [file, setFile] = useState(null)
  const [mode, setMode] = useState("auto")
  const [busy, setBusy] = useState(false)
  const [jobId, setJobId] = useState(null)

  async function handle(e){
    e.preventDefault()
    if(!file) return alert("Choose a file")
    setBusy(true)
    setStatus && setStatus('uploading')
    const fd = new FormData()
    fd.append("file", file)
    const up = await fetch("/api/upload",{method:"POST",body:fd})
    const j = await up.json()
    setStatus && setStatus('uploaded')
    // submit job
    const resp = await fetch("/api/submit",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({filename:j.filename, mode})
    })
    const rj = await resp.json()
    if(resp.ok && rj.job_id){
      setJobId(rj.job_id)
      setStatus && setStatus('queued')
      // poll for status
      const poll = setInterval(async ()=>{
        const s = await fetch(`/api/status/${rj.job_id}`).then(res=>res.json()).catch(()=>null)
        if(!s) return
        setStatus && setStatus(s.status)
        if(s.status === "done" || s.status === "error"){
          clearInterval(poll)
          setBusy(false)
          if(s.status === "done"){
            onDone && onDone(s.output)
          } else {
            alert("Job error: " + s.error)
          }
        }
      }, 1500)
    } else {
      alert("Submit failed: " + (rj.detail || JSON.stringify(rj)))
      setBusy(false)
    }
  }

  return (
    <form onSubmit={handle} className="upload-form">
      <input type="file" accept=".mid,.midi,.mp3,.wav" onChange={e=>setFile(e.target.files[0])}/>
      <select onChange={e=>setMode(e.target.value)} value={mode}>
        <option value="auto">Auto</option>
        <option value="music">Music Improve</option>
        <option value="audio">Audio Upscale</option>
      </select>
      <button type="submit" disabled={busy}>{busy?("Processing... " + (jobId?("job "+jobId):"")):"Upload & Improve"}</button>
    </form>
  )
}
