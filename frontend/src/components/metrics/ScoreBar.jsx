import React from 'react'

export function ScoreBar({ value, max, color }) {
    const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0
    const fill = color === 'green' ? 'var(--apple-green)' : color === 'yellow' ? 'var(--apple-yellow)' : 'var(--apple-red)'
    return (
        <div className="score-bar">
            <div className="score-bar-fill" style={{ width: `${pct}%`, background: fill }} />
        </div>
    )
}
