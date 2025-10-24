import React, { useState } from 'react'

export default function UploadForm({ onDone, setStatus }) {
  const [file, setFile] = useState(null)
  const [mode, setMode] = useState("improve")
  const [busy, setBusy] = useState(false)
  const [jobId, setJobId] = useState(null)

  async function handleSubmit(e){
    e.preventDefault()
    if(!file) return alert("Choose a file first")
    setBusy(true)
    const fd = new FormData()
    fd.append("file", file)
    const up = await fetch("/api/upload", {method:"POST", body:fd})
    const j = await up.json()
    const endpoint = mode === "train" ? "/api/train" : "/api/submit"

    const resp = await fetch(endpoint, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({filename: j.filename})
    })
    const rj = await resp.json()
    if(resp.ok && rj.job_id){
      setJobId(rj.job_id)
      setStatus && setStatus('queued')
      const poll = setInterval(async ()=>{
        const s = await fetch(`/api/status/${rj.job_id}`).then(r=>r.json()).catch(()=>null)
        if(!s) return
        setStatus(s.status)
        if(s.status === "done" || s.status === "error"){
          clearInterval(poll)
          setBusy(false)
          if(s.status === "done"){
            alert(mode === "train" ? "Training complete!" : "Improvement done!")
            onDone && onDone(s.output)
          } else {
            alert("Job error: " + s.error)
          }
        }
      }, 2000)
    } else {
      alert("Submit failed: " + (rj.detail || JSON.stringify(rj)))
      setBusy(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="upload-form">
      <input type="file" accept=".mid,.midi,.mp3,.wav" onChange={e=>setFile(e.target.files[0])}/>
      <select onChange={e=>setMode(e.target.value)} value={mode}>
        <option value="improve">Improve Music</option>
        <option value="train">Train Model</option>
      </select>
      <button type="submit" disabled={busy}>
        {busy ? `Processing (${mode}) ...` : mode === "train" ? "Train Model" : "Improve Music"}
      </button>
    </form>
  )
}

