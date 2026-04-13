/**
 * MessageBubble.tsx — Renders a single chat message.
 *
 * User messages: right-aligned pill (light gray background)
 * Assistant messages: full-width block with answer text, source cards, confidence badge
 */

import { clsx } from 'clsx'
import { FileText, Cpu } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '../types'

/** Normalise LLM output to clean markdown before rendering.
 *  Small models often use • or · as bullets instead of markdown `- `.
 *  This converts them so ReactMarkdown renders proper <ul> lists. */
function normaliseMarkdown(text: string): string {
  return text
    // • or · at start of line → markdown list item
    .replace(/^[•·]\s*/gm, '- ')
    // inline • separating items on same line → split onto new lines
    .replace(/\s[•·]\s/g, '\n- ')
}


interface Props {
  message: Message
}

export default function MessageBubble({ message }: Props) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] bg-raised border border-border rounded-2xl rounded-tr-sm
                        px-4 py-2.5 text-sm text-black leading-relaxed whitespace-pre-wrap">
          {message.content}
        </div>
      </div>
    )
  }

  // ── Assistant message ──────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-3 max-w-[85%]">

      {/* Answer text — rendered as markdown */}
      <div
        className={clsx(
          'prose prose-sm max-w-none text-black',
          // Tune the prose defaults to match the app's design tokens
          'prose-headings:font-semibold prose-headings:text-black',
          'prose-strong:text-black prose-strong:font-semibold',
          'prose-em:text-black',
          'prose-code:text-black prose-code:bg-raised prose-code:rounded prose-code:px-1 prose-code:py-0.5 prose-code:text-xs prose-code:font-mono prose-code:before:content-none prose-code:after:content-none',
          'prose-pre:bg-raised prose-pre:border prose-pre:border-border prose-pre:rounded-lg prose-pre:text-xs',
          'prose-table:text-sm prose-th:text-left prose-th:font-semibold prose-th:text-black prose-td:text-black',
          'prose-thead:border-border prose-tr:border-border',
          'prose-a:text-black prose-a:underline',
          'prose-li:text-black prose-ul:my-1 prose-ol:my-1',
          'prose-p:my-1 prose-p:leading-relaxed',
          'prose-blockquote:border-l-border prose-blockquote:text-muted',
          // Streaming states
          message.isStreaming && !message.content && 'text-muted italic',
          message.isStreaming && message.content && 'streaming-cursor',
        )}
      >
        {message.isStreaming && !message.content
          ? <p className="text-muted italic not-prose">Thinking…</p>
          : <ReactMarkdown remarkPlugins={[remarkGfm]}>{normaliseMarkdown(message.content)}</ReactMarkdown>
        }
      </div>

      {/* Source cards — shown after streaming completes */}
      {!message.isStreaming && message.sources.length > 0 && (
        <SourceList message={message} />
      )}
    </div>
  )
}

// ── Source citation list ───────────────────────────────────────────────────────

function SourceList({ message }: { message: Message }) {
  const { sources, confidence, model_used } = message

  return (
    <div className="flex flex-col gap-2">
      {/* Source cards */}
      <div className="flex flex-wrap gap-2">
        {sources.map((src, i) => (
          <div
            key={i}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border
                       bg-raised text-xs text-muted"
          >
            <FileText size={11} className="shrink-0 text-black/40" />
            <span className="font-medium text-black truncate max-w-[180px]" title={src.filename}>
              {src.filename}
            </span>
            <span>p.{src.page}</span>
            {src.was_ocr && (
              <span className="text-muted border border-border rounded px-1">OCR</span>
            )}
            <span className="text-dim">{(src.score * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>

      {/* Confidence + model row */}
      <div className="flex items-center gap-3">
        {confidence && (
          <span
            className={clsx(
              'text-xs px-2 py-0.5 rounded-full border font-medium',
              confidence.level === 'HIGH'   && 'border-green-300  text-green-700 bg-green-50',
              confidence.level === 'MEDIUM' && 'border-yellow-300 text-yellow-700 bg-yellow-50',
              confidence.level === 'LOW'    && 'border-red-300    text-red-700   bg-red-50',
            )}
            title={confidence.reason}
          >
            {confidence.level}
          </span>
        )}
        {model_used && (
          <span className="flex items-center gap-1 text-xs text-dim">
            <Cpu size={10} />
            {model_used}
          </span>
        )}
      </div>
    </div>
  )
}
