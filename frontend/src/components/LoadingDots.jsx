import React from 'react'

export function LoadingDots() {
    return (
        <div className="flex justify-start mb-5">
            <div className="glass rounded-[20px] rounded-bl-md px-5 py-3.5">
                <div className="loading-dots"><span></span><span></span><span></span></div>
            </div>
        </div>
    )
}
