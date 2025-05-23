import { useState } from 'react'

export default function Chat() {
  const [text, setText] = useState('')
  const [messages, setMessages] = useState<string[]>([])

  async function send() {
    const res = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: 'demo', message: text })
    })
    const data = await res.json()
    setMessages(data.history)
    setText('')
  }

  return (
    <div className="p-8">
      <h1 className="text-xl font-bold mb-4">Chat</h1>
      <div className="space-y-2 mb-4">
        {messages.map((m,i) => <div key={i}>{m}</div>)}
      </div>
      <input className="border p-2 mr-2" value={text} onChange={e => setText(e.target.value)} />
      <button className="bg-green-500 text-white px-4 py-2" onClick={send}>Send</button>
    </div>
  )
}
