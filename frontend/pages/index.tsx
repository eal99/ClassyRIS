import Link from 'next/link'

export default function Home() {
  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">ClassyRIS Hub</h1>
      <ul className="space-y-2">
        <li><Link href="/search">Search</Link></li>
        <li><Link href="/imagesearch">Image Search</Link></li>
        <li><Link href="/analytics">Analytics</Link></li>
        <li><Link href="/chat">Chat</Link></li>
        <li><Link href="/hub">Employee Hub</Link></li>
      </ul>
    </div>
  )
}
