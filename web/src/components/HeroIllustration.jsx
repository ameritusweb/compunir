import React from 'react'

const HeroIllustration = () => {
  return (
    <svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
      {/* Background gradient */}
      <defs>
        <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style={{stopColor:"#1a1a2e",stopOpacity:1}} />
          <stop offset="100%" style={{stopColor:"#2d2d44",stopOpacity:1}} />
        </linearGradient>
        
        {/* Glow effects */}
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>

        {/* Rainbow trail */}
        <linearGradient id="rainbow" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style={{stopColor:"#ff1b6b",stopOpacity:0.8}}/>
          <stop offset="50%" style={{stopColor:"#45caff",stopOpacity:0.8}}/>
          <stop offset="100%" style={{stopColor:"#7d48ff",stopOpacity:0.8}}/>
        </linearGradient>
      </defs>

      {/* Animated particles */}
      <g>
        <circle cx="600" cy="100" r="2" fill="#45caff" filter="url(#glow)">
          <animate attributeName="cy" values="100;120;100" dur="2s" repeatCount="indefinite"/>
        </circle>
        <circle cx="650" cy="150" r="2" fill="#ff1b6b" filter="url(#glow)">
          <animate attributeName="cy" values="150;170;150" dur="1.5s" repeatCount="indefinite"/>
        </circle>
        <circle cx="700" cy="200" r="2" fill="#7d48ff" filter="url(#glow)">
          <animate attributeName="cy" values="200;220;200" dur="2.5s" repeatCount="indefinite"/>
        </circle>
      </g>
    </svg>
  )
}

export default HeroIllustration