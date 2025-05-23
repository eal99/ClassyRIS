import { useEffect, useState } from 'react'

export default function Analytics() {
  const [summary, setSummary] = useState<any>(null)

  useEffect(() => {
    fetch('http://localhost:8000/analytics/summary')
      .then(res => res.json())
      .then(data => setSummary(data))
      .catch(() => setSummary(null))
  }, [])

  return (
    <div className="p-8">
      <h1 className="text-xl font-bold mb-4">Analytics</h1>
      {summary ? (
        <div className="space-y-2">
          <div>Total products: {summary.total_products}</div>
          {summary.average_price && <div>Average price: ${summary.average_price.toFixed(2)}</div>}
        </div>
      ) : (
        <div>Loading...</div>
      )}
    </div>
  )
}
