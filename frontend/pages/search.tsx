import { useState } from 'react'

export default function Search() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<any[]>([])

  async function handleSearch() {
    const res = await fetch('http://localhost:8000/search/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 5 })
    })
    setResults(await res.json())
  }

  return (
    <div className="p-8">
      <h1 className="text-xl font-bold mb-4">Search</h1>
      <input className="border p-2 mr-2" value={query} onChange={e => setQuery(e.target.value)} />
      <button className="bg-blue-500 text-white px-4 py-2" onClick={handleSearch}>Search</button>
      <pre className="mt-4 bg-gray-100 p-2">
        {JSON.stringify(results, null, 2)}
      </pre>
    </div>
  )
}
