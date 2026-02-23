import React, { useState, useRef } from 'react'
import { METRIC_TIPS } from './MetricTips'

export function Tip({ id, children }) {
    const text = METRIC_TIPS[id]
    if (!text) return children
    const [tipPos, setTipPos] = useState(null)
    const ref = useRef(null)
    const show = () => {
        if (ref.current) {
            const r = ref.current.getBoundingClientRect()
            setTipPos({ top: r.top, left: r.left + r.width / 2 })
        }
    }
    return (
        <span ref={ref} className="metric-tip" onMouseEnter={show} onMouseLeave={() => setTipPos(null)}>
            {children}
            {tipPos && (
                <span style={{
                    position: 'fixed',
                    top: tipPos.top - 6,
                    left: tipPos.left,
                    transform: 'translateX(-50%) translateY(-100%)',
                    background: '#1f2937',
                    color: '#d1d5db',
                    fontSize: '11px',
                    fontWeight: 400,
                    lineHeight: 1.4,
                    padding: '6px 10px',
                    borderRadius: '8px',
                    border: '1px solid #374151',
                    whiteSpace: 'normal',
                    width: 'max-content',
                    maxWidth: '260px',
                    zIndex: 9999,
                    pointerEvents: 'none',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
                }}>
                    {text}
                </span>
            )}
        </span>
    )
}
