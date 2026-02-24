import React, { useState, useRef } from 'react'
import ReactDOM from 'react-dom'
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
            {tipPos && ReactDOM.createPortal(
                <span style={{
                    position: 'fixed',
                    top: tipPos.top - 6,
                    left: tipPos.left,
                    transform: 'translateX(-50%) translateY(-100%)',
                    background: 'var(--apple-bg-elevated)',
                    color: 'var(--apple-text-primary)',
                    fontSize: '11px',
                    fontWeight: 400,
                    lineHeight: 1.4,
                    padding: '6px 10px',
                    borderRadius: '8px',
                    border: '1px solid var(--apple-border-primary)',
                    whiteSpace: 'normal',
                    width: 'max-content',
                    maxWidth: '260px',
                    zIndex: 9999,
                    pointerEvents: 'none',
                    boxShadow: 'var(--apple-shadow-md)',
                }}>
                    {text}
                </span>,
                document.body
            )}
        </span>
    )
}
