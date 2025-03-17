import { useState } from 'react'
import { Theme } from '@radix-ui/themes'
import LandingPage from './components/LandingPage'
import './App.css'

function App() {
  return (
    <Theme appearance="dark" accentColor="blue" grayColor="slate" scaling="95%" radius="medium">
      <LandingPage />
    </Theme>
  )
}

export default App
