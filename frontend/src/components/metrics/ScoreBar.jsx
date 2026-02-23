import React from 'react'

export function ScoreBar({ value, max, color }) {
    const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0
    const fill = color === 'green' ? '#4ade80' : color === 'yellow' ? '#fbbf24' : '#f87171'
    return (
        <div className="score-bar">
            <div className="score-bar-fill" style={{ width: `${pct}%`, background: fill }} />
        </div>
    )
}
