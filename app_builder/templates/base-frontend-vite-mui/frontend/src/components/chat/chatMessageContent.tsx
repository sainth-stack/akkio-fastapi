import { Fragment } from 'react';
import { Typography } from '@mui/material';

/** Minimal markdown: **bold** and bullet lines for assistant replies. */
export function ChatMessageContent({ content, isUser }: { content: string; isUser: boolean }) {
  if (isUser) {
    return (
      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
        {content}
      </Typography>
    );
  }

  const lines = content.split('\n');
  return (
    <Typography component="div" variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
      {lines.map((line, i) => (
        <Fragment key={i}>
          <LineWithBold text={line} />
          {i < lines.length - 1 ? '\n' : null}
        </Fragment>
      ))}
    </Typography>
  );
}

function LineWithBold({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <>
      {parts.map((part, idx) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return (
            <Typography
              key={idx}
              component="span"
              variant="body2"
              sx={{ fontWeight: 700 }}
            >
              {part.slice(2, -2)}
            </Typography>
          );
        }
        return <span key={idx}>{part}</span>;
      })}
    </>
  );
}
