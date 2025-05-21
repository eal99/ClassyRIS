import { useState } from 'react'

export default function ImageSearch() {
  const [file, setFile] = useState<File | null>(null)
  const [results, setResults] = useState<any[]>([])

  async function handleSearch() {
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    const res = await fetch('http://localhost:8000/search/image?top_k=5', {
      method: 'POST',
      body: form
    })
    setResults(await res.json())
  }

  return (
    <div className="p-8">
      <h1 className="text-xl font-bold mb-4">Image Search</h1>
      <input type="file" accept="image/*" onChange={e => setFile(e.target.files?.[0] || null)} />
      <button className="bg-blue-500 text-white px-4 py-2 ml-2" onClick={handleSearch} disabled={!file}>Search</button>
      <pre className="mt-4 bg-gray-100 p-2">
        {JSON.stringify(results, null, 2)}
      </pre>
    </div>
  )
}
