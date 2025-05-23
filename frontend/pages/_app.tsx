import type { AppProps } from 'next/app'
import '../styles.css'
import { CssBaseline } from '@mui/material'

export default function MyApp({ Component, pageProps }: AppProps) {
  return (
    <>
      <CssBaseline />
      <Component {...pageProps} />
    </>
  )
}
