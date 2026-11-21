import React, { useState, useEffect, useRef } from 'react';
import './index.css';
import Spinner from 'react-bootstrap/Spinner';
import { Collapse, Tag, Progress, Button, message as antdMessage, Tooltip as AntdTooltip, Modal, Dropdown, Menu, Tabs, Input, Empty, Switch } from 'antd';
import { FaCheckCircle, FaBrain, FaDownload, FaTrashAlt, FaInfoCircle, FaThermometerHalf, FaPlus, FaEllipsisV, FaFolderPlus, FaSearch, FaMicrophone } from 'react-icons/fa';
import api, { wsUrl, wsAuthPayload } from '../../utils/api';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import Plot from 'react-plotly.js';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import mammoth from 'mammoth';
import MultiModelTraining from './MultiModelTraining';
import AgentTemplates from './AgentTemplates';
import PremiumOverlay from '../../components/PremiumOverlay';

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

// Utility function to clean HTML content
const cleanHtmlContent = (html) => {
  if (!html) return html;

  let cleaned = html;

  // Remove literal \n characters (not the actual newlines in the source, but the string "\n")
  cleaned = cleaned.replace(/\\n/g, '');

  // Remove all <br>, <br/>, and <br /> tags
  cleaned = cleaned.replace(/<br\s*\/?>/gi, '');

  // Remove excessive whitespace between tags
  cleaned = cleaned.replace(/>\s+</g, '><');

  // Trim whitespace
  cleaned = cleaned.trim();

  return cleaned;
};

const formatFieldLabel = (key) => {
  return String(key)
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\b\w/g, (c) => c.toUpperCase());
};

const tryParseJsonAnswer = (text) => {
  if (!text || typeof text !== 'string') return null;
  const trimmed = text.trim();
  if (!trimmed) return null;

  const attempts = [
    trimmed,
    trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1]?.trim(),
    trimmed.match(/\{[\s\S]*\}/)?.[0],
    trimmed.match(/\[[\s\S]*\]/)?.[0],
  ].filter(Boolean);

  for (const candidate of attempts) {
    try {
      return JSON.parse(candidate);
    } catch {
      // continue
    }
  }
  return null;
};

const PLATFORM_STYLES = {
  instagram: { bg: 'linear-gradient(135deg, #833ab4, #fd1d1d, #fcb045)', label: 'Instagram' },
  linkedin: { bg: '#0a66c2', label: 'LinkedIn' },
  twitter: { bg: '#0f1419', label: 'Twitter / X' },
  x: { bg: '#0f1419', label: 'X' },
  facebook: { bg: '#1877f2', label: 'Facebook' },
  tiktok: { bg: '#010101', label: 'TikTok' },
  youtube: { bg: '#ff0000', label: 'YouTube' },
};

const isSocialPostData = (obj) => {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return false;
  const text = obj.content || obj.post || obj.caption || obj.text || obj.body;
  return Boolean(text) && Boolean(obj.platform || obj.hashtags || obj.suggested_image_prompt || obj.image_prompt);
};

const SocialPostCard = ({ data }) => {
  const platformKey = String(data.platform || '').toLowerCase();
  const platformStyle = PLATFORM_STYLES[platformKey] || { bg: '#6366f1', label: data.platform || 'Social Post' };
  const postText = data.content || data.post || data.caption || data.text || data.body || '';
  const hashtags = data.hashtags || data.tags || '';
  const imagePrompt = data.suggested_image_prompt || data.image_prompt || data.visual_prompt || '';

  const hashtagList = typeof hashtags === 'string'
    ? hashtags.split(/\s+/).filter((t) => t.startsWith('#'))
    : Array.isArray(hashtags) ? hashtags : [];

  return (
    <div className="social-post-card">
      <div className="social-post-card__header">
        <span className="social-post-card__platform" style={{ background: platformStyle.bg }}>
          {platformStyle.label}
        </span>
        <span className="social-post-card__type">Generated post</span>
      </div>
      <div className="social-post-card__body">
        <p className="social-post-card__content">{postText}</p>
        {hashtagList.length > 0 && (
          <div className="social-post-card__tags">
            {hashtagList.map((tag, i) => (
              <span key={i} className="social-post-card__tag">{tag}</span>
            ))}
          </div>
        )}
      </div>
      {imagePrompt && (
        <div className="social-post-card__prompt">
          <div className="social-post-card__prompt-label">Suggested image prompt</div>
          <p>{imagePrompt}</p>
        </div>
      )}
    </div>
  );
};

const StructuredDataCard = ({ data }) => {
  if (Array.isArray(data)) {
    if (data.every((item) => typeof item === 'string')) {
      return (
        <ul className="structured-data-list">
          {data.map((item, i) => <li key={i}>{item}</li>)}
        </ul>
      );
    }
    return (
      <div className="structured-data-stack">
        {data.map((item, i) => (
          <div key={i} className="structured-data-card structured-data-card--nested">
            <StructuredDataCard data={item} />
          </div>
        ))}
      </div>
    );
  }

  if (typeof data !== 'object' || data === null) {
    return <p className="structured-data-text">{String(data)}</p>;
  }

  const entries = Object.entries(data).filter(([, v]) => v !== null && v !== undefined && v !== '');

  return (
    <div className="structured-data-card">
      {entries.map(([key, value]) => (
        <div key={key} className="structured-data-row">
          <div className="structured-data-row__label">{formatFieldLabel(key)}</div>
          <div className="structured-data-row__value">
            {typeof value === 'object' ? (
              <StructuredDataCard data={value} />
            ) : (
              <span>{String(value)}</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

const AnswerBody = ({ answer, isError = false, isStreaming = false }) => {
  if (!answer) return null;

  const parsed = tryParseJsonAnswer(answer);

  if (parsed !== null) {
    if (isSocialPostData(parsed)) {
      return <SocialPostCard data={parsed} />;
    }
    return <StructuredDataCard data={parsed} />;
  }

  const looksLikeHtml = typeof answer === 'string' && /<[a-z][\s\S]*>/i.test(answer);

  if (isStreaming && answer.trim().startsWith('{') && !looksLikeHtml) {
    return <div className="thinking-indicator">Composing response...</div>;
  }

  return (
    <div className={`answer-body${isError ? ' answer-body--error' : ''}`}>
      {looksLikeHtml ? (
        <div className="answer-body__html" dangerouslySetInnerHTML={{ __html: cleanHtmlContent(answer) }} />
      ) : (
        <p className="answer-body__text">{answer}</p>
      )}
    </div>
  );
};

// Component to render plan text with clickable links
const PlanContent = ({ planText }) => {
  if (!planText) return null;

  // Split text by lines and process each line
  const lines = planText.split('\n');

  return (
    <div>
      {lines.map((line, lineIdx) => {
        // Check if line contains a URL
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        const parts = [];
        let lastIndex = 0;
        let match;

        while ((match = urlRegex.exec(line)) !== null) {
          // Add text before URL
          if (match.index > lastIndex) {
            parts.push({ type: 'text', content: line.substring(lastIndex, match.index) });
          }

          // Add URL as link
          parts.push({ type: 'link', content: match[0], url: match[0] });
          lastIndex = match.index + match[0].length;
        }

        // Add remaining text
        if (lastIndex < line.length) {
          parts.push({ type: 'text', content: line.substring(lastIndex) });
        }

        // If no URLs found, just render the line
        if (parts.length === 0) {
          return (
            <div key={lineIdx}>
              {line || '\u00A0'}
            </div>
          );
        }

        // Render line with links
        return (
          <div key={lineIdx}>
            {parts.map((part, partIdx) => {
              if (part.type === 'link') {
                return (
                  <a
                    key={partIdx}
                    href={part.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      color: '#2563eb',
                      textDecoration: 'none',
                      borderBottom: '1px solid #2563eb'
                    }}
                    onMouseEnter={(e) => e.target.style.textDecoration = 'underline'}
                    onMouseLeave={(e) => e.target.style.textDecoration = 'none'}
                  >
                    {part.content}
                  </a>
                );
              }
              return <span key={partIdx}>{part.content}</span>;
            })}
          </div>
        );
      })}
    </div>
  );
};

// PDF Viewer Component using react-pdf
const PDFViewer = ({ fileData: pdfDataUrl, filename }) => {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [loading, setLoading] = useState(true);

  function onDocumentLoadSuccess({ numPages }) {
    setNumPages(numPages);
    setLoading(false);
  }

  function onDocumentLoadError(error) {
    console.error('PDF Load Error:', error);
    setLoading(false);
  }

  return (
    <div>
      {loading && (
        <div style={{ padding: '2rem', textAlign: 'center' }}>
          <Spinner animation="border" size="sm" />
          <p style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: '#6b7280' }}>Loading PDF...</p>
        </div>
      )}
      <Document
        file={pdfDataUrl}
        onLoadSuccess={onDocumentLoadSuccess}
        onLoadError={onDocumentLoadError}
        loading=""
      >
        <Page
          pageNumber={pageNumber}
          renderTextLayer={true}
          renderAnnotationLayer={true}
          width={350}
        />
      </Document>
      {numPages && (
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '0.75rem',
          backgroundColor: '#f9fafb',
          borderTop: '1px solid #e5e7eb',
          fontSize: '0.875rem'
        }}>
          <button
            onClick={() => setPageNumber(prev => Math.max(1, prev - 1))}
            disabled={pageNumber <= 1}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: pageNumber <= 1 ? '#e5e7eb' : '#2563eb',
              color: pageNumber <= 1 ? '#9ca3af' : 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: pageNumber <= 1 ? 'not-allowed' : 'pointer',
              fontSize: '0.875rem',
              fontWeight: '600'
            }}
          >
            Previous
          </button>
          <span style={{ color: '#374151', fontWeight: '500' }}>
            Page {pageNumber} of {numPages}
          </span>
          <button
            onClick={() => setPageNumber(prev => Math.min(numPages, prev + 1))}
            disabled={pageNumber >= numPages}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: pageNumber >= numPages ? '#e5e7eb' : '#2563eb',
              color: pageNumber >= numPages ? '#9ca3af' : 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: pageNumber >= numPages ? 'not-allowed' : 'pointer',
              fontSize: '0.875rem',
              fontWeight: '600'
            }}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
};

// Word Document Viewer using mammoth
const WordViewer = ({ fileData: docxDataUrl, filename }) => {
  const [htmlContent, setHtmlContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadDocument = async () => {
      try {
        setLoading(true);
        // Convert data URL to ArrayBuffer
        const base64Data = docxDataUrl.split(',')[1];
        const binaryString = atob(base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }

        // Use mammoth to convert to HTML
        const result = await mammoth.convertToHtml({ arrayBuffer: bytes.buffer });
        setHtmlContent(result.value);
        setLoading(false);
      } catch (err) {
        console.error('Word document conversion error:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    if (docxDataUrl) {
      loadDocument();
    }
  }, [docxDataUrl]);

  if (loading) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center' }}>
        <Spinner animation="border" size="sm" />
        <p style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: '#6b7280' }}>Loading Word document...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '1rem', color: '#dc2626', backgroundColor: '#fee2e2', borderRadius: '4px' }}>
        <p style={{ margin: 0, fontSize: '0.875rem' }}>Failed to load document: {error}</p>
      </div>
    );
  }

  return (
    <div
      style={{
        padding: '1.5rem',
        backgroundColor: '#ffffff',
        maxHeight: '70vh',
        overflow: 'auto',
        fontSize: '0.95rem',
        lineHeight: '1.8',
        color: '#1f2937'
      }}
      dangerouslySetInnerHTML={{ __html: cleanHtmlContent(htmlContent) }}
    />
  );
};

// Component to render file preview based on type
const FilePreview = ({ fileData, filename }) => {
  if (!fileData) {
    return (
      <div style={{ padding: '1rem', textAlign: 'center', color: '#6b7280' }}>
        No file preview available
      </div>
    );
  }

  const { type, preview_data, file_data, text_content, transcription } = fileData;

  // Debug log
  console.log('[FilePreview] Data:', { type, hasPreviewData: !!preview_data, hasFileData: !!file_data, hasTextContent: !!text_content, textContentLength: text_content?.length });

  // CSV/Excel/Tabular table preview
  if ((type === 'csv' || type === 'excel' || type === 'tabular') && preview_data) {
    const { columns, rows, total_rows, preview_rows } = preview_data;
    return (
      <div style={{ padding: '1rem' }}>
        <div style={{ marginBottom: '1rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '0.5rem' }}>
          <AntdTooltip title={filename} placement="top">
            <h3 style={{
              fontSize: '1rem',
              fontWeight: '600',
              margin: 0,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              maxWidth: '100%',
              cursor: 'default'
            }}>
              {filename}
            </h3>
          </AntdTooltip>
          <p style={{ fontSize: '0.875rem', color: '#6b7280', margin: '0.25rem 0 0 0' }}>
            {total_rows} rows × {columns.length} columns
            {preview_rows < total_rows && ` (showing first ${preview_rows})`}
          </p>
        </div>
        <div style={{ overflow: 'auto', maxHeight: '70vh', border: '1px solid #e5e7eb', borderRadius: '4px' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
            <thead style={{ position: 'sticky', top: 0, backgroundColor: '#f9fafb', zIndex: 10 }}>
              <tr>
                {columns.map((col, idx) => (
                  <th
                    key={idx}
                    style={{
                      padding: '0.5rem',
                      textAlign: 'left',
                      borderBottom: '2px solid #e5e7eb',
                      fontWeight: '600',
                      fontSize: '0.75rem',
                      textTransform: 'uppercase',
                      color: '#374151'
                    }}
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIdx) => (
                <tr key={rowIdx} style={{ backgroundColor: rowIdx % 2 === 0 ? '#ffffff' : '#f9fafb' }}>
                  {columns.map((col, colIdx) => (
                    <td
                      key={colIdx}
                      style={{
                        padding: '0.5rem',
                        borderBottom: '1px solid #f3f4f6',
                        color: '#111827',
                        maxWidth: '200px',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}
                      title={String(row[col] || '')}
                    >
                      {String(row[col] || '')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // PDF preview using react-pdf (handle both 'pdf' and 'document' type with PDF data)
  if (type === 'pdf' || (type === 'document' && file_data && file_data.includes('application/pdf'))) {
    return (
      <div style={{ padding: '1rem' }}>
        <div style={{ marginBottom: '1rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <AntdTooltip title={filename} placement="top">
            <h3 style={{
              fontSize: '1rem',
              fontWeight: '600',
              margin: 0,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              maxWidth: '100%',
              cursor: 'default'
            }}>
              {filename}
            </h3>
          </AntdTooltip>
          {file_data && (
            <a
              href={file_data}
              download={filename}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.5rem 1rem',
                backgroundColor: '#059669',
                color: 'white',
                textDecoration: 'none',
                borderRadius: '6px',
                fontSize: '0.875rem',
                fontWeight: '600'
              }}
            >
              ⬇ Download
            </a>
          )}
        </div>
        {file_data ? (
          <div style={{ border: '1px solid #e5e7eb', borderRadius: '4px', overflow: 'hidden', backgroundColor: '#f9fafb' }}>
            <PDFViewer fileData={file_data} filename={filename} />
          </div>
        ) : (
          <div style={{ padding: '2rem', textAlign: 'center', color: '#6b7280', backgroundColor: '#f9fafb', borderRadius: '4px' }}>
            <p style={{ margin: 0 }}>No preview available for this PDF.</p>
          </div>
        )}
      </div>
    );
  }

  // Word document preview using mammoth (handle both 'word' and 'document' type with Word data)
  if (type === 'word' || (type === 'document' && file_data && file_data.includes('wordprocessingml'))) {
    return (
      <div style={{ padding: '1rem' }}>
        <div style={{ marginBottom: '1rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <AntdTooltip title={filename} placement="top">
            <h3 style={{
              fontSize: '1rem',
              fontWeight: '600',
              margin: 0,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              maxWidth: '100%',
              cursor: 'default'
            }}>
              {filename}
            </h3>
          </AntdTooltip>
          {file_data && (
            <a
              href={file_data}
              download={filename}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.5rem 1rem',
                backgroundColor: '#2563eb',
                color: 'white',
                textDecoration: 'none',
                borderRadius: '6px',
                fontSize: '0.875rem',
                fontWeight: '600'
              }}
            >
              ⬇ Download
            </a>
          )}
        </div>
        {file_data ? (
          <div style={{ border: '1px solid #e5e7eb', borderRadius: '4px', overflow: 'hidden' }}>
            <WordViewer fileData={file_data} filename={filename} />
          </div>
        ) : text_content ? (
          <div style={{ padding: '1rem', backgroundColor: '#ffffff', borderRadius: '4px', maxHeight: '70vh', overflow: 'auto', border: '1px solid #e5e7eb' }}>
            <div style={{ fontSize: '0.875rem', color: '#374151', whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
              {text_content}
            </div>
          </div>
        ) : (
          <div style={{ padding: '2rem', textAlign: 'center', color: '#6b7280', backgroundColor: '#f9fafb', borderRadius: '4px' }}>
            <p style={{ margin: 0 }}>No preview available for this document.</p>
          </div>
        )}
      </div>
    );
  }

  // XML preview
  if (type === 'xml') {
    return (
      <div style={{ padding: '1rem' }}>
        <div style={{ marginBottom: '1rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '0.5rem' }}>
          <AntdTooltip title={filename} placement="top">
            <h3 style={{
              fontSize: '1rem',
              fontWeight: '600',
              margin: 0,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              maxWidth: '100%',
              cursor: 'default'
            }}>
              {filename}
            </h3>
          </AntdTooltip>
        </div>
        {text_content ? (
          <div style={{ padding: '1rem', backgroundColor: '#1f2937', borderRadius: '4px', maxHeight: '70vh', overflow: 'auto', border: '1px solid #374151' }}>
            <pre style={{
              fontSize: '0.8rem',
              color: '#f3f4f6',
              margin: 0,
              whiteSpace: 'pre-wrap',
              fontFamily: 'Monaco, Menlo, "Courier New", monospace',
              lineHeight: '1.5'
            }}>
              {text_content}
            </pre>
          </div>
        ) : file_data ? (
          <div style={{ height: '70vh', border: '1px solid #e5e7eb', borderRadius: '4px', overflow: 'hidden' }}>
            <iframe
              src={file_data}
              style={{ width: '100%', height: '100%', border: 'none' }}
              title="XML Preview"
            />
          </div>
        ) : (
          <div style={{ padding: '2rem', textAlign: 'center', color: '#6b7280', backgroundColor: '#f9fafb', borderRadius: '4px' }}>
            <p style={{ margin: 0 }}>No XML preview available.</p>
          </div>
        )}
      </div>
    );
  }

  // Image preview
  if (type === 'image') {
    return (
      <div style={{ padding: '1rem' }}>
        <div style={{ marginBottom: '1rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '0.5rem' }}>
          <AntdTooltip title={filename} placement="top">
            <h3 style={{
              fontSize: '1rem',
              fontWeight: '600',
              margin: 0,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              maxWidth: '100%',
              cursor: 'default'
            }}>
              {filename}
            </h3>
          </AntdTooltip>
        </div>
        {file_data ? (
          <div style={{
            textAlign: 'center',
            border: '1px solid #e5e7eb',
            borderRadius: '4px',
            padding: '1rem',
            backgroundColor: '#f9fafb',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '200px'
          }}>
            <img
              src={file_data}
              alt={filename}
              style={{
                maxWidth: '100%',
                maxHeight: '70vh',
                objectFit: 'contain',
                borderRadius: '4px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
              }}
            />
          </div>
        ) : (
          <div style={{ padding: '2rem', textAlign: 'center', color: '#6b7280', backgroundColor: '#f9fafb', borderRadius: '4px' }}>
            <p style={{ margin: 0 }}>No image preview available.</p>
          </div>
        )}
      </div>
    );
  }

  // Audio preview
  if (type === 'audio' && file_data) {
    return (
      <div style={{ padding: '1rem' }}>
        <div style={{ marginBottom: '1rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '0.5rem' }}>
          <AntdTooltip title={filename} placement="top">
            <h3 style={{
              fontSize: '1rem',
              fontWeight: '600',
              margin: 0,
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              maxWidth: '100%',
              cursor: 'default'
            }}>
              {filename}
            </h3>
          </AntdTooltip>
        </div>
        <div style={{ marginBottom: '1rem', padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '4px' }}>
          <audio controls style={{ width: '100%' }}>
            <source src={file_data} />
            Your browser does not support the audio element.
          </audio>
        </div>
        {transcription && (
          <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '4px', maxHeight: '40vh', overflow: 'auto' }}>
            <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>Transcription:</h4>
            <p style={{ fontSize: '0.875rem', color: '#374151', whiteSpace: 'pre-wrap', margin: 0 }}>
              {transcription}
            </p>
          </div>
        )}
      </div>
    );
  }

  // Fallback for other types or no preview
  return (
    <div style={{ padding: '1rem', textAlign: 'center', color: '#6b7280' }}>
      <p>Preview not available for this file type: {type}</p>
      {text_content && (
        <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '4px', textAlign: 'left', maxHeight: '60vh', overflow: 'auto' }}>
          <pre style={{ fontSize: '0.875rem', color: '#374151', margin: 0, whiteSpace: 'pre-wrap' }}>
            {text_content}
          </pre>
        </div>
      )}
    </div>
  );
};

// Component to render report content
const ReportContent = ({ data, generationFormat }) => {
  const [isDownloading, setIsDownloading] = useState(false);
  const [isSavingReport, setIsSavingReport] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const reportRef = useRef(null);

  if (!data?.report || Object.keys(data.report).length === 0) return null;
  const { heading, paragraphs, table, charts, analysis_charts, forecasting_charts } = data.report;

  // Determine mode
  // If generationFormat is present and NOT "null" string, and NOT null value, then we are in explicit mode
  const fmt = (generationFormat && generationFormat !== 'null') ? generationFormat.toLowerCase() : null;
  const isModePdf = fmt && fmt.includes('pdf');
  const isModeCsv = fmt && (fmt.includes('csv') || fmt.includes('excel'));
  const isModeNormal = !isModePdf && !isModeCsv;
  const isModeExplicit = isModePdf || isModeCsv;

  const generatePDF = async () => {
    console.log(reportRef)

    if (!reportRef.current) return null;

    try {
      setIsDownloading(true);

      // Create a new jsPDF instance
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = pdf.internal.pageSize.getHeight();

      // Capture the report content with optimized quality for clear text and smaller file size
      const canvas = await html2canvas(reportRef.current, {
        scale: 2, // Balanced scale for good text clarity without excessive file size
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#ffffff',
        width: reportRef.current.scrollWidth,
        height: reportRef.current.scrollHeight,
        logging: false, // Disable logging for performance
        imageTimeout: 15000, // 15 second timeout for images
        windowWidth: reportRef.current.scrollWidth,
        windowHeight: reportRef.current.scrollHeight,
        onclone: (clonedDoc) => {
          // Optimize fonts and styles for better rendering
          const clonedElement = clonedDoc.querySelector('.report-content') || clonedDoc.body;
          if (clonedElement) {
            clonedElement.style.fontSize = '14px';
            clonedElement.style.lineHeight = '1.6';
          }
        }
      });

      // Use JPEG with high quality (0.95) for good text clarity and smaller file size
      // JPEG at 95% quality provides excellent text clarity while being much smaller than PNG
      const imgData = canvas.toDataURL('image/jpeg', 0.95);
      const imgWidth = pdfWidth - 20; // 10mm margin on each side
      const imgHeight = (canvas.height * imgWidth) / canvas.width;

      let position = 10; // Start 10mm from top

      // Add content to PDF
      if (imgHeight <= pdfHeight - 20) {
        // Content fits on one page
        pdf.addImage(imgData, 'JPEG', 10, position, imgWidth, imgHeight, undefined, 'FAST');
      } else {
        // Content spans multiple pages
        let remainingHeight = imgHeight;
        let currentPosition = 0;

        while (remainingHeight > 0) {
          const pageHeight = Math.min(remainingHeight, pdfHeight - 20);
          const canvasHeight = (pageHeight * canvas.height) / imgHeight;

          // Create a temporary canvas for this page
          const pageCanvas = document.createElement('canvas');
          pageCanvas.width = canvas.width;
          pageCanvas.height = canvasHeight;
          const pageCtx = pageCanvas.getContext('2d');

          pageCtx.drawImage(
            canvas,
            0, currentPosition,
            canvas.width, canvasHeight,
            0, 0,
            canvas.width, canvasHeight
          );

          // Use JPEG with high quality (0.95) for page images
          const pageImgData = pageCanvas.toDataURL('image/jpeg', 0.95);
          pdf.addImage(pageImgData, 'JPEG', 10, 10, imgWidth, pageHeight, undefined, 'FAST');

          remainingHeight -= pageHeight;
          currentPosition += canvasHeight;

          if (remainingHeight > 0) {
            pdf.addPage();
          }
        }
      }

      return pdf;
    } catch (error) {
      console.error('Error generating PDF:', error);
      alert('Error generating PDF. Please try again.');
      return null;
    } finally {
      setIsDownloading(false);
    }
  };

  const downloadPDF = async () => {
    const pdf = await generatePDF();
    if (pdf) {
      const fileName = `${heading?.replace(/[^a-zA-Z0-9]/g, '_') || 'Report'}_${new Date().toISOString().split('T')[0]}.pdf`;
      pdf.save(fileName);
    }
  };



  const renderPlotlyChart = (chart, index) => {
    if (!chart.plotly) return null;

    return (
      <div key={index} className="chart-container" style={{
        marginBottom: '2rem',
        backgroundColor: '#ffffff',
        padding: '1.5rem',
        borderRadius: '8px',
        border: '1px solid #e5e7eb'
      }}>
        <h4 style={{
          marginBottom: '1.5rem',
          color: '#1f2937',
          fontSize: '1.125rem',
          fontWeight: '600',
          textAlign: 'center',
          borderBottom: '1px solid #e5e7eb',
          paddingBottom: '0.75rem'
        }}>
          {chart.title}
        </h4>
        <div style={{ height: '450px', width: '100%' }}>
          <Plot
            data={chart.plotly.data}
            layout={{
              ...chart.plotly.layout,
              autosize: true,
              font: {
                family: 'Inter, system-ui, sans-serif',
                size: 12,
                color: '#374151'
              },
              plot_bgcolor: '#ffffff',
              paper_bgcolor: '#ffffff',
              margin: { l: 60, r: 40, t: 40, b: 60 },
              showlegend: true,
              legend: {
                orientation: 'h',
                x: 0.5,
                xanchor: 'center',
                y: -0.15
              },
              grid: {
                rows: 1,
                columns: 1,
                pattern: 'independent'
              }
            }}
            config={{
              responsive: true,
              displayModeBar: true,
              modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
              displaylogo: false
            }}
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </div>
    );
  };

  const renderChart = (chart, index) => {
    // Handle the data format from your JSON
    let chartData;
    let seriesNames;

    if (chart.data && Array.isArray(chart.data)) {
      // Transform the data to the expected format
      chartData = chart.data.reduce((acc, item) => {
        const existingItem = acc.find(accItem => accItem.x === item.x);
        if (existingItem) {
          existingItem[item.series] = item.y;
        } else {
          acc.push({ x: item.x, [item.series]: item.y });
        }
        return acc;
      }, []);

      seriesNames = [...new Set(chart.data.map(item => item.series))];
    } else {
      // Fallback for other data formats
      chartData = chart.data || [];
      seriesNames = chart.series || [];
    }

    const colors = chart.config?.colors || ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];

    if (chart.chartType === 'bar') {
      return (
        <div key={index} className="chart-container" style={{
          marginBottom: '2rem',
          backgroundColor: '#ffffff',
          padding: '1.5rem',
          borderRadius: '8px',
          border: '1px solid #e5e7eb'
        }}>
          <h4 style={{
            marginBottom: '1.5rem',
            color: '#1f2937',
            fontSize: '1.125rem',
            fontWeight: '600',
            textAlign: 'center',
            borderBottom: '1px solid #e5e7eb',
            paddingBottom: '0.75rem'
          }}>
            {chart.title}
          </h4>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis
                dataKey="x"
                tick={{ fontSize: 12, fill: '#6b7280' }}
                axisLine={{ stroke: '#d1d5db' }}
              />
              <YAxis
                tick={{ fontSize: 12, fill: '#6b7280' }}
                axisLine={{ stroke: '#d1d5db' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '4px'
                }}
              />
              <Legend />
              {seriesNames.map((series, seriesIndex) => (
                <Bar
                  key={series}
                  dataKey={series}
                  fill={colors[seriesIndex % colors.length]}
                  radius={[2, 2, 0, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      );
    } else if (chart.chartType === 'line') {
      return (
        <div key={index} className="chart-container" style={{
          marginBottom: '2rem',
          backgroundColor: '#ffffff',
          padding: '1.5rem',
          borderRadius: '8px',
          border: '1px solid #e5e7eb'
        }}>
          <h4 style={{
            marginBottom: '1.5rem',
            color: '#1f2937',
            fontSize: '1.125rem',
            fontWeight: '600',
            textAlign: 'center',
            borderBottom: '1px solid #e5e7eb',
            paddingBottom: '0.75rem'
          }}>
            {chart.title}
          </h4>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis
                dataKey="x"
                tick={{ fontSize: 12, fill: '#6b7280' }}
                axisLine={{ stroke: '#d1d5db' }}
              />
              <YAxis
                tick={{ fontSize: 12, fill: '#6b7280' }}
                axisLine={{ stroke: '#d1d5db' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '4px'
                }}
              />
              <Legend />
              {seriesNames.map((series, seriesIndex) => (
                <Line
                  key={series}
                  type="monotone"
                  dataKey={series}
                  stroke={colors[seriesIndex % colors.length]}
                  strokeWidth={3}
                  dot={{ r: 4 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="report-wrapper" style={{ width: '100%', maxWidth: 'none' }}>
      {/* Action Buttons */}


      {/* Report Content - This is what gets captured for PDF */}
      <div ref={reportRef} className="report-content" style={{
        backgroundColor: '#ffffff',
        padding: '2rem',
        borderRadius: '8px',
        border: '1px solid #e5e7eb',
        fontSize: '14px',
        lineHeight: '1.6',
        color: '#374151'
      }}>
        {/* Heading */}
        {heading && (
          <h1 style={{
            fontSize: '2rem',
            fontWeight: '700',
            color: '#1f2937',
            marginBottom: '2rem',
            textAlign: 'center',
            borderBottom: '2px solid #e5e7eb',
            paddingBottom: '1rem'
          }}>
            {heading}
          </h1>
        )}

        {/* Paragraphs */}
        {paragraphs && paragraphs.length > 0 && (
          <div style={{ marginBottom: '2rem' }}>
            {paragraphs.map((paragraph, index) => (
              <div
                key={index}
                style={{
                  marginBottom: '1rem',
                  color: '#374151',
                  lineHeight: '1.8'
                }}
                dangerouslySetInnerHTML={{ __html: cleanHtmlContent(paragraph) }}
              />
            ))}
          </div>
        )}

        {/* Table */}
        {table && table.headers && table.headers.length > 0 && (
          <div style={{ marginBottom: '2rem', overflow: 'auto' }}>
            <table style={{
              width: '100%',
              borderCollapse: 'collapse',
              border: '1px solid #e5e7eb'
            }}>
              <thead>
                <tr style={{ backgroundColor: '#f9fafb' }}>
                  {table.headers.map((header, idx) => (
                    <th key={idx} style={{
                      padding: '0.75rem',
                      textAlign: 'left',
                      borderBottom: '2px solid #e5e7eb',
                      fontWeight: '600',
                      color: '#374151'
                    }}>
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows && table.rows.map((row, rowIdx) => (
                  <tr key={rowIdx} style={{
                    backgroundColor: rowIdx % 2 === 0 ? '#ffffff' : '#f9fafb'
                  }}>
                    {row.map((cell, cellIdx) => (
                      <td key={cellIdx} style={{
                        padding: '0.75rem',
                        borderBottom: '1px solid #e5e7eb',
                        color: '#374151'
                      }}>
                        {String(cell || '')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Charts */}
        {charts && charts.length > 0 && charts.map((chart, index) => (
          chart.plotly ? renderPlotlyChart(chart, index) : renderChart(chart, index)
        ))}

        {/* Analysis Charts */}
        {analysis_charts && analysis_charts.length > 0 && (
          <div style={{ marginTop: '2rem' }}>
            <h3 style={{
              fontSize: '1.5rem',
              fontWeight: '600',
              color: '#1f2937',
              marginBottom: '1rem'
            }}>
              Analysis
            </h3>
            {analysis_charts.map((chart, index) => (
              chart.plotly ? renderPlotlyChart(chart, index) : renderChart(chart, index)
            ))}
          </div>
        )}

        {/* Forecasting Charts */}
        {forecasting_charts && forecasting_charts.length > 0 && (
          <div style={{ marginTop: '2rem' }}>
            <h3 style={{
              fontSize: '1.5rem',
              fontWeight: '600',
              color: '#1f2937',
              marginBottom: '1rem'
            }}>
              Forecasting
            </h3>
            {forecasting_charts.map((chart, index) => (
              chart.plotly ? renderPlotlyChart(chart, index) : renderChart(chart, index)
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const GeneralAnswer = ({ data, isError = false }) => {
  if (!data?.answer) return null;
  return <AnswerBody answer={data.answer} isError={isError} />;
};

// Component to render simple text/explanation answers
const AnswerText = ({ payload, explanation }) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      {explanation && (
        <div style={{
          background: '#eef2ff',
          border: '1px solid #c7d2fe',
          color: '#3730a3',
          padding: '0.5rem 0.75rem',
          borderRadius: 6,
          fontSize: 13,
        }} dangerouslySetInnerHTML={{ __html: cleanHtmlContent(explanation) }}>
          {/* {explanation} */}
        </div>
      )}
      <div style={{ color: '#111827', fontSize: 15, lineHeight: 1.6, whiteSpace: 'pre-wrap' }} dangerouslySetInnerHTML={{ __html: cleanHtmlContent(payload) }} />
    </div>
  );
};

// Component to render table/list results
const AnswerTable = ({ rows, explanation }) => {
  if (!Array.isArray(rows) || rows.length === 0) {
    return <AnswerText payload="No rows to display." explanation={explanation} />;
  }

  // Build a stable set of columns from the union of keys across rows
  const firstRowKeys = Object.keys(rows[0] || {});
  const extraKeys = Array.from(
    rows.reduce((set, r) => {
      Object.keys(r || {}).forEach((k) => set.add(k));
      return set;
    }, new Set())
  ).filter((k) => !firstRowKeys.includes(k));
  const columns = [...firstRowKeys, ...extraKeys];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      {explanation && (
        <div style={{
          background: '#ecfeff',
          border: '1px solid #a5f3fc',
          color: '#155e75',
          padding: '0.5rem 0.75rem',
          borderRadius: 6,
          fontSize: 13,
        }} dangerouslySetInnerHTML={{ __html: cleanHtmlContent(explanation) }}>
          {/* {explanation} */}
        </div>
      )}
      <div style={{ width: '100%', overflow: 'auto', border: '1px solid #e5e7eb', borderRadius: 8 }}>
        <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 0 }}>
          <thead>
            <tr style={{ backgroundColor: '#f9fafb' }}>
              {columns.map((col) => (
                <th key={col} style={{
                  position: 'sticky',
                  top: 0,
                  textAlign: 'left',
                  padding: '10px 12px',
                  borderBottom: '1px solid #e5e7eb',
                  fontWeight: 600,
                  fontSize: 12,
                  color: '#374151',
                  whiteSpace: 'nowrap',
                }}>
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIdx) => (
              <tr key={rowIdx} style={{ backgroundColor: rowIdx % 2 === 0 ? '#ffffff' : '#f9fafb' }}>
                {columns.map((col) => (
                  <td key={col} style={{
                    padding: '10px 12px',
                    borderBottom: '1px solid #f3f4f6',
                    fontSize: 13,
                    color: '#111827',
                    maxWidth: 420,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}>
                    {row && row[col] !== undefined && row[col] !== null
                      ? String(row[col])
                      : ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// Component to render Plotly chart from figure dict
const AnswerPlotly = ({ figure, explanation }) => {
  const rawData = (figure && (figure.data || figure.figure?.data)) || [];
  const rawLayout = (figure && (figure.layout || figure.figure?.layout)) || {};

  // Normalize traces: ensure arrays exist and lengths match; allow categorical y for scatter/line; drop empty traces
  let missingYWarning = false;
  const MAX_POINTS_PER_TRACE = 10000; // hard cap for rendering performance
  const WEBGL_THRESHOLD = 50000; // switch to scattergl beyond this size

  const toEpoch = (v) => {
    const t = typeof v === 'string' || v instanceof Date ? Date.parse(v) : (Number.isFinite(v) ? v : NaN);
    return Number.isFinite(t) ? t : NaN;
  };

  // Largest-Triangle-Three-Buckets downsampling for numeric series
  const downsampleLTTB = (xArr, yArr, threshold) => {
    const length = xArr.length;
    if (threshold >= length || threshold <= 2) return { x: xArr, y: yArr };

    // Convert x to numeric domain for area calculations
    const xNum = xArr.map((v, i) => {
      const n = toEpoch(v);
      return Number.isFinite(n) ? n : i; // fallback to index if not a date/number
    });

    const sampledX = new Array(threshold);
    const sampledY = new Array(threshold);

    // Bucket size. Leave room for first and last data points
    const bucketSize = (length - 2) / (threshold - 2);
    let a = 0; // initially a is the first point
    sampledX[0] = xArr[0];
    sampledY[0] = yArr[0];

    for (let i = 0; i < threshold - 2; i++) {
      // range for this bucket
      const rangeStart = Math.floor((i + 1) * bucketSize) + 1;
      const rangeEnd = Math.floor((i + 2) * bucketSize) + 1;
      const rangeEndClamped = Math.min(rangeEnd, length);

      // Calculate average for next bucket (used for calculating area)
      let avgX = 0;
      let avgY = 0;
      const avgRangeStart = rangeStart;
      const avgRangeEnd = rangeEndClamped;
      const avgRangeLength = avgRangeEnd - avgRangeStart;
      for (let j = avgRangeStart; j < avgRangeEnd; j++) {
        avgX += xNum[j];
        avgY += Number(yArr[j]) || 0;
      }
      avgX /= Math.max(1, avgRangeLength);
      avgY /= Math.max(1, avgRangeLength);

      // Get the point from this bucket that forms the largest triangle
      let maxArea = -1;
      let maxIndex = rangeStart;
      for (let j = rangeStart; j < rangeEndClamped; j++) {
        const area = Math.abs(
          (xNum[a] - avgX) * (Number(yArr[j]) - Number(yArr[a])) -
          (xNum[a] - xNum[j]) * (avgY - Number(yArr[a]))
        );
        if (area > maxArea) {
          maxArea = area;
          maxIndex = j;
        }
      }
      sampledX[i + 1] = xArr[maxIndex];
      sampledY[i + 1] = yArr[maxIndex];
      a = maxIndex; // next a is this bucket's chosen point
    }

    sampledX[threshold - 1] = xArr[length - 1];
    sampledY[threshold - 1] = yArr[length - 1];
    return { x: sampledX, y: sampledY };
  };

  const strideSample = (xArr, yArr, threshold) => {
    const length = xArr.length;
    if (length <= threshold) return { x: xArr, y: yArr };
    const stride = Math.ceil(length / threshold);
    const sx = [];
    const sy = [];
    for (let i = 0; i < length; i += stride) {
      sx.push(xArr[i]);
      sy.push(yArr[i]);
    }
    // Ensure last point is included
    if (sx[sx.length - 1] !== xArr[length - 1]) {
      sx.push(xArr[length - 1]);
      sy.push(yArr[length - 1]);
    }
    return { x: sx, y: sy };
  };

  const normalizedData = (Array.isArray(rawData) ? rawData : [])
    .map((trace) => {
      const x = Array.isArray(trace?.x) ? trace.x : [];
      let yCandidate = Array.isArray(trace?.y) ? trace.y : (Array.isArray(trace?.values) ? trace.values : []);
      if ((!yCandidate || yCandidate.length === 0) && Array.isArray(x) && x.length > 0 && (trace?.type === 'scatter' || trace?.type === 'lines')) {
        yCandidate = x.map((_, i) => i + 1);
        missingYWarning = true;
      }

      const isScatterLike = (
        trace?.type === 'scatter' ||
        trace?.type === 'scattergl' ||
        (typeof trace?.mode === 'string' && (trace.mode.includes('lines') || trace.mode.includes('markers')))
      );

      const y = isScatterLike
        ? yCandidate.map((v) => (v === null || v === undefined || v === '' ? null : v))
        : yCandidate
          .map((v) => (v === null || v === undefined || v === '' ? null : Number(v)))
          .map((v) => (Number.isFinite(v) ? v : null));

      const filtered = x.reduce((acc, xv, i) => {
        const yv = y[i];
        if (isScatterLike) {
          if (yv !== null) {
            acc.x.push(xv);
            acc.y.push(yv);
          }
        } else if (yv !== null && Number.isFinite(yv)) {
          acc.x.push(xv);
          acc.y.push(yv);
        }
        return acc;
      }, { x: [], y: [] });

      // Downsample large traces
      const pointCount = filtered.x.length;
      let ds = { x: filtered.x, y: filtered.y };
      if (pointCount > MAX_POINTS_PER_TRACE) {
        const yIsNumeric = filtered.y.every((v) => typeof v === 'number' && Number.isFinite(v));
        ds = yIsNumeric ? downsampleLTTB(filtered.x, filtered.y, MAX_POINTS_PER_TRACE) : strideSample(filtered.x, filtered.y, MAX_POINTS_PER_TRACE);
      }

      return {
        ...trace,
        type: (isScatterLike && pointCount > WEBGL_THRESHOLD) ? 'scattergl' : trace?.type,
        x: ds.x,
        y: ds.y,
        hovertemplate: trace?.hovertemplate || '%{x}<br>%{y}<extra></extra>',
        marker: {
          ...(trace?.marker || {}),
          line: { width: 0 },
        },
      };
    })
    .filter((t) => Array.isArray(t.x) && t.x.length > 0 && Array.isArray(t.y) && t.y.length > 0);

  if (!normalizedData.length) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        {explanation && (
          <div style={{
            background: '#fef3c7',
            border: '1px solid #fde68a',
            color: '#7c2d12',
            padding: '0.5rem 0.75rem',
            borderRadius: 6,
            fontSize: 13,
          }} dangerouslySetInnerHTML={{ __html: cleanHtmlContent(explanation) }}>
            {/* {explanation} */}
          </div>
        )}
        <div style={{
          width: '100%',
          padding: '16px',
          border: '1px dashed #e5e7eb',
          borderRadius: 8,
          background: '#fafafa',
          color: '#6b7280',
          fontSize: 14,
          textAlign: 'center',
        }}>
          No chart data to display.
        </div>
      </div>
    );
  }

  const isCategoricalY = normalizedData.some((t) => (t?.y || []).some((v) => typeof v === 'string'));

  const layout = {
    ...rawLayout,
    autosize: true,
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    margin: { l: 60, r: 30, t: 48, b: 80, ...(rawLayout?.margin || {}) },
    font: { family: 'Inter, system-ui, sans-serif', size: 12, color: '#374151', ...(rawLayout?.font || {}) },
    showlegend: rawLayout?.showlegend ?? true,
    legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.2, ...(rawLayout?.legend || {}) },
    xaxis: { ...(rawLayout?.xaxis || {}), tickangle: rawLayout?.xaxis?.tickangle ?? -45, gridcolor: '#f3f4f6', zerolinecolor: '#e5e7eb' },
    yaxis: { ...(rawLayout?.yaxis || {}), gridcolor: '#f3f4f6', zerolinecolor: '#e5e7eb', type: isCategoricalY ? 'category' : (rawLayout?.yaxis?.type || undefined) },
    template: undefined, // avoid heavy default templates that clash with app theme
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      {explanation && (
        <div style={{
          background: '#fef3c7',
          border: '1px solid #fde68a',
          color: '#7c2d12',
          padding: '0.5rem 0.75rem',
          borderRadius: 6,
          fontSize: 13,
        }} dangerouslySetInnerHTML={{ __html: cleanHtmlContent(explanation) }}>
          {/* {explanation} */}
        </div>
      )}
      {missingYWarning && (
        <div style={{
          background: '#fff7ed',
          border: '1px solid #fed7aa',
          color: '#9a3412',
          padding: '0.4rem 0.65rem',
          borderRadius: 6,
          fontSize: 12,
        }}>
          Y values were missing in the response; plotted index values as a fallback.
        </div>
      )}
      <div style={{ width: '100%', height: 420 }}>
        <Plot
          data={normalizedData}
          layout={layout}
          config={{ responsive: true, displaylogo: false, displayModeBar: 'hover', modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d'] }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
};

// Component to render Multi-Model Answer
const MultiModelAnswer = ({ data }) => {
  const { answer, confidence, agents_used, sources, reasoning, validation_notes, model_name, report } = data;
  const { Panel } = Collapse;

  return (
    <div className="multi-model-answer">
      <AnswerBody answer={answer} />

      {/* {report && (
        <div style={{ marginBottom: 20, borderTop: '1px solid #e5e7eb', paddingTop: 20 }}>
          {(!data.generation_format || data.generation_format === 'null') && (
            <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8, color: '#1f2937', display: 'flex', alignItems: 'center', gap: 8 }}>
              <FaBrain color="#3b82f6" />
              Generated Report
            </div>
          )}
          <ReportContent
            data={{ report, title: report.heading, description: `Report generated on ${new Date().toLocaleDateString()}` }}
            generationFormat={data.generation_format}
          />
        </div>
      )} */}

      {/* Project Structure Display */}
      {data.project_structure && data.project_structure.files && (
        <div style={{ marginBottom: 20, borderTop: '1px solid #e5e7eb', paddingTop: 20 }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: '#1f2937', display: 'flex', alignItems: 'center', gap: 8 }}>
            <FaCheckCircle color="#10b981" />
            Generated Project: {data.project_structure.project_title}
          </div>

          {data.deployment_url && (
            <div style={{
              marginBottom: 16,
              padding: 12,
              background: '#f0fdf4',
              border: '1px solid #bbf7d0',
              borderRadius: 6,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#15803d' }}>Deployed Successfully</div>
                <a href={data.deployment_url} target="_blank" rel="noopener noreferrer" style={{ fontSize: 13, color: '#16a34a', textDecoration: 'underline' }}>
                  {data.deployment_url}
                </a>
              </div>
            </div>
          )}

          <Collapse accordion>
            {data.project_structure.files.map((file, idx) => (
              <Panel
                header={<span style={{ fontFamily: 'monospace', fontWeight: 500 }}>{file.path}</span>}
                key={idx}
              >
                <div style={{
                  background: '#1e293b',
                  color: '#f1f5f9',
                  padding: 16,
                  borderRadius: 6,
                  fontSize: 12,
                  fontFamily: 'Monaco, Consolas, "Courier New", monospace',
                  maxHeight: 400,
                  overflow: 'auto',
                  whiteSpace: 'pre'
                }}>
                  {file.content}
                </div>
              </Panel>
            ))}
          </Collapse>
        </div>
      )}

      {confidence !== undefined && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8, color: '#1f2937' }}>
            Confidence Score
          </div>
          <Progress
            percent={Math.round(confidence * 100)}
            strokeColor={{
              '0%': '#ef4444',
              '50%': '#f59e0b',
              '100%': '#22c55e'
            }}
          />
        </div>
      )}

      {/* {agents_used && agents_used.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8, color: '#1f2937' }}>
            AI Agents Used
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {agents_used.map((agent, idx) => (
              <Tag key={idx} color="blue">
                {agent}
              </Tag>
            ))}
          </div>
        </div>
      )} */}
      {/* 
      {sources && sources.length > 0 && (
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8, color: '#1f2937' }}>
            Sources ({sources.length})
          </div>
          <Collapse accordion>
            {sources.map((source, idx) => (
              <Panel
                header={
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <Tag color={source.file_type === 'tabular' ? 'blue' : source.file_type === 'document' ? 'red' : 'green'}>
                      {source.file_type}
                    </Tag>
                    <span style={{ fontWeight: 600 }}>{source.file_name}</span>
                    {source.relevance && (
                      <Tag color="orange">
                        {Math.round(source.relevance * 100)}% relevant
                      </Tag>
                    )}
                  </div>
                }
                key={idx}
              >
                {source.excerpt && (
                  <div style={{
                    background: '#f3f4f6',
                    padding: 12,
                    borderRadius: 6,
                    fontSize: 13,
                    fontFamily: 'monospace',
                    whiteSpace: 'pre-wrap',
                    color: '#374151'
                  }}>
                    {source.excerpt}
                  </div>
                )}
              </Panel>
            ))}
          </Collapse>
        </div>
      )}

      {reasoning && (
        <div style={{ marginTop: 20 }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8, color: '#1f2937' }}>
            Reasoning Process
          </div>
          <div style={{
            background: 'white',
            padding: 12,
            borderRadius: 8,
            fontSize: 13,
            color: '#6b7280',
            fontStyle: 'italic'
          }}>
            {reasoning}
          </div>
        </div>
      )}

      {validation_notes && (
        <div style={{ marginTop: 20 }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8, color: '#1f2937' }}>
            Validation Notes
          </div>
          <div style={{
            background: 'white',
            padding: 12,
            borderRadius: 8,
            fontSize: 13,
            color: '#6b7280',
            borderLeft: '3px solid #3b82f6'
          }}>
            {validation_notes}
          </div>
        </div>
      )}

      {model_name && (
        <div style={{ marginTop: 20, textAlign: 'right' }}>
          <Tag color="purple">
            Model: {model_name}
          </Tag>
        </div>
      )} */}
    </div>
  );
};

// Helper to build tree from flat paths
const buildFileTree = (files) => {
  const root = {};
  files.forEach(file => {
    const parts = file.path.split('/');
    let current = root;
    parts.forEach((part, index) => {
      if (!current[part]) {
        current[part] = {
          name: part,
          path: parts.slice(0, index + 1).join('/'),
          type: index === parts.length - 1 ? 'file' : 'folder',
          children: {},
          content: index === parts.length - 1 ? file.content : null
        };
      }
      current = current[part].children;
    });
  });
  return root;
};

// Recursive Tree Item Component
const FileTreeItem = ({ node, level = 0, onFileClick, viewingFile }) => {
  const [expanded, setExpanded] = useState(true);
  const hasChildren = node.type === 'folder' && Object.keys(node.children).length > 0;
  const isSelected = viewingFile && viewingFile.path === node.path;

  return (
    <div style={{ paddingLeft: `${level * 12}px` }}>
      <div
        onClick={(e) => {
          e.stopPropagation();
          if (node.type === 'folder') {
            setExpanded(!expanded);
          } else {
            onFileClick(node);
          }
        }}
        style={{
          padding: '6px 8px',
          borderRadius: '4px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          color: isSelected ? '#2563eb' : '#374151',
          backgroundColor: isSelected ? '#eff6ff' : 'transparent',
          fontSize: '0.85rem',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          transition: 'background-color 0.1s'
        }}
        onMouseEnter={(e) => {
          if (!isSelected) e.currentTarget.style.backgroundColor = '#f3f4f6';
        }}
        onMouseLeave={(e) => {
          if (!isSelected) e.currentTarget.style.backgroundColor = 'transparent';
        }}
      >
        <span style={{ fontSize: '14px', color: '#9ca3af', width: '16px', display: 'flex', justifyContent: 'center' }}>
          {node.type === 'folder' ? (expanded ? '📂' : '📁') : '📄'}
        </span>
        <span style={{
          fontWeight: node.type === 'folder' ? 600 : 400,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap'
        }}>
          {node.name}
        </span>
      </div>

      {node.type === 'folder' && expanded && (
        <div>
          {Object.values(node.children)
            .sort((a, b) => {
              if (a.type === b.type) return a.name.localeCompare(b.name);
              return a.type === 'folder' ? -1 : 1;
            })
            .map((child) => (
              <FileTreeItem
                key={child.path}
                node={child}
                level={level + 1}
                onFileClick={onFileClick}
                viewingFile={viewingFile}
              />
            ))}
        </div>
      )}
    </div>
  );
};

// Component to render Multi-Model Files List with Preview
const MultiModelFilesList = ({ files, sessionId, userEmail, generatedDocuments = [], projectFiles = [], onProjectFileClick, onDeploy, viewingFile }) => {
  const { Panel } = Collapse;
  const [fileData, setFileData] = useState({});
  const [loadingFiles, setLoadingFiles] = useState({});
  const [downloadingFiles, setDownloadingFiles] = useState({});
  const [activeKeys, setActiveKeys] = useState([]); // No panels open by default
  const [activeProjectKeys, setActiveProjectKeys] = useState([]);

  // Download file function
  const handleDownload = React.useCallback(async (file) => {
    const fileName = file.file_name;

    // Set downloading state
    setDownloadingFiles(prev => ({ ...prev, [fileName]: true }));

    try {
      // Handle generated documents (especially reports)
      if (file.is_generated) {
        if (file.report_data) {
          // Generate PDF from report data using html2canvas
          // Create a temporary container to render the report
          const tempContainer = document.createElement('div');
          tempContainer.style.cssText = 'position: absolute; left: -9999px; width: 800px; padding: 2rem; background: white; font-family: Arial, sans-serif;';
          document.body.appendChild(tempContainer);

          // Render report HTML
          let htmlContent = '';
          if (file.report_data.heading) {
            htmlContent += `<h1 style="font-size: 2rem; font-weight: 700; text-align: center; margin-bottom: 2rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 1rem;">${file.report_data.heading}</h1>`;
          }

          if (file.report_data.paragraphs) {
            file.report_data.paragraphs.forEach(para => {
              htmlContent += `<div style="margin-bottom: 1rem; line-height: 1.8; color: #374151;">${para}</div>`;
            });
          }

          if (file.report_data.table && file.report_data.table.headers) {
            htmlContent += '<table style="width: 100%; border-collapse: collapse; margin: 2rem 0; border: 1px solid #e5e7eb;">';
            htmlContent += '<thead><tr style="background-color: #f9fafb;">';
            file.report_data.table.headers.forEach(header => {
              htmlContent += `<th style="padding: 0.75rem; text-align: left; border-bottom: 2px solid #e5e7eb; font-weight: 600;">${header}</th>`;
            });
            htmlContent += '</tr></thead><tbody>';
            if (file.report_data.table.rows) {
              file.report_data.table.rows.forEach((row, idx) => {
                htmlContent += `<tr style="background-color: ${idx % 2 === 0 ? '#ffffff' : '#f9fafb'};">`;
                row.forEach(cell => {
                  htmlContent += `<td style="padding: 0.75rem; border-bottom: 1px solid #e5e7eb;">${String(cell || '')}</td>`;
                });
                htmlContent += '</tr>';
              });
            }
            htmlContent += '</tbody></table>';
          }

          tempContainer.innerHTML = htmlContent;

          // Wait for content to render
          await new Promise(resolve => setTimeout(resolve, 100));

          // Generate PDF using html2canvas with optimized quality
          const canvas = await html2canvas(tempContainer, {
            scale: 2, // Balanced scale for good text clarity without excessive file size
            useCORS: true,
            allowTaint: true,
            backgroundColor: '#ffffff',
            logging: false,
            windowWidth: tempContainer.scrollWidth,
            windowHeight: tempContainer.scrollHeight,
            onclone: (clonedDoc) => {
              // Optimize fonts for better rendering
              const clonedElement = clonedDoc.body;
              if (clonedElement) {
                clonedElement.style.fontSize = '14px';
                clonedElement.style.lineHeight = '1.6';
              }
            }
          });

          // Use JPEG with high quality (0.95) for good text clarity and smaller file size
          const imgData = canvas.toDataURL('image/jpeg', 0.95);
          const pdf = new jsPDF('p', 'mm', 'a4');
          const pdfWidth = pdf.internal.pageSize.getWidth();
          const pdfHeight = pdf.internal.pageSize.getHeight();
          const imgWidth = pdfWidth - 20;
          const imgHeight = (canvas.height * imgWidth) / canvas.width;

          let position = 10;
          if (imgHeight <= pdfHeight - 20) {
            pdf.addImage(imgData, 'JPEG', 10, position, imgWidth, imgHeight);
          } else {
            let remainingHeight = imgHeight;
            let currentPosition = 0;
            while (remainingHeight > 0) {
              const pageHeight = Math.min(remainingHeight, pdfHeight - 20);
              const canvasHeight = (pageHeight * canvas.height) / imgHeight;
              const pageCanvas = document.createElement('canvas');
              pageCanvas.width = canvas.width;
              pageCanvas.height = canvasHeight;
              const pageCtx = pageCanvas.getContext('2d');
              pageCtx.drawImage(canvas, 0, currentPosition, canvas.width, canvasHeight, 0, 0, canvas.width, canvasHeight);
              const pageImgData = pageCanvas.toDataURL('image/jpeg', 0.95);
              pdf.addImage(pageImgData, 'JPEG', 10, 10, imgWidth, pageHeight, undefined, 'FAST');
              remainingHeight -= pageHeight;
              currentPosition += canvasHeight;
              if (remainingHeight > 0) {
                pdf.addPage();
              }
            }
          }

          pdf.save(fileName);
          document.body.removeChild(tempContainer);
          setDownloadingFiles(prev => ({ ...prev, [fileName]: false }));
          return;
        } else if (file.file_data) {
          // Direct file data available
          if (file.file_data.startsWith('data:')) {
            const response = await fetch(file.file_data);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = fileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
            setDownloadingFiles(prev => ({ ...prev, [fileName]: false }));
            return;
          } else {
            const link = document.createElement('a');
            link.href = file.file_data;
            link.download = fileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setDownloadingFiles(prev => ({ ...prev, [fileName]: false }));
            return;
          }
        }
      }

      // Fetch file data for uploaded files
      const response = await api.get(`/multi-model/file-preview`, {
        params: {
          session_id: sessionId,
          file_name: fileName,
          
        }
      });

      const filePreviewData = response.data;
      const { type, file_data, text_content, preview_data } = filePreviewData;

      // Handle different file types
      if (file_data) {
        // If file_data is a data URL or URL, use it directly
        if (file_data.startsWith('data:')) {
          // Data URL - convert to blob and download
          const response = await fetch(file_data);
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = fileName;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          window.URL.revokeObjectURL(url);
        } else {
          // Regular URL - create download link
          const link = document.createElement('a');
          link.href = file_data;
          link.download = fileName;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }
      } else if (type === 'csv' || type === 'excel' || type === 'tabular') {
        // Handle CSV/Excel files - convert preview data to CSV
        if (preview_data && preview_data.columns && preview_data.rows) {
          const csvContent = [
            preview_data.columns.join(','),
            ...preview_data.rows.map(row =>
              preview_data.columns.map(col => {
                const value = String(row[col] || '');
                return `"${value.replace(/"/g, '""')}"`;
              }).join(',')
            )
          ].join('\n');

          const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = fileName.endsWith('.csv') ? fileName : `${fileName}.csv`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
        }
      } else if (text_content) {
        // Handle text files
        const blob = new Blob([text_content], { type: 'text/plain;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      } else {
        antdMessage.warning('File download not available for this file type');
      }
    } catch (error) {
      console.error(`Error downloading file ${fileName}:`, error);
      antdMessage.error('Failed to download file. Please try again.');
    } finally {
      setDownloadingFiles(prev => ({ ...prev, [fileName]: false }));
    }
  }, [sessionId, userEmail]);

  // Fetch file preview when panel is opened
  const handlePanelChange = React.useCallback(async (keys) => {
    setActiveKeys(keys);

    // Find newly opened panels
    const newKeys = keys.filter(key => !activeKeys.includes(key));

    for (const key of newKeys) {
      const idx = parseInt(key);
      const allFiles = [
        ...(files || []).map(f => ({ ...f, is_generated: false })),
        ...(generatedDocuments || []).map(f => ({ ...f, is_generated: true }))
      ];
      if (!allFiles || !allFiles[idx]) continue;

      const file = allFiles[idx];

      // Skip if already loaded
      if (fileData[file.file_name]) continue;

      // For generated documents with preview data, use it directly
      if (file.is_generated && file.file_preview_data) {
        setFileData(prev => ({ ...prev, [file.file_name]: file.file_preview_data }));
        continue;
      }

      // For generated reports, we don't need to fetch preview
      if (file.is_generated && file.report_data) {
        setFileData(prev => ({ ...prev, [file.file_name]: { type: 'document', report_data: file.report_data } }));
        continue;
      }

      // Set loading state for uploaded files
      setLoadingFiles(prev => ({ ...prev, [file.file_name]: true }));

      try {
        const response = await api.get(`/multi-model/file-preview`, {
          params: {
            session_id: sessionId,
            file_name: file.file_name,
            
          }
        });

        setFileData(prev => ({ ...prev, [file.file_name]: response.data }));
      } catch (error) {
        console.error(`Error loading preview for ${file.file_name}:`, error);
        setFileData(prev => ({
          ...prev,
          [file.file_name]: { error: 'Failed to load preview' }
        }));
      } finally {
        setLoadingFiles(prev => ({ ...prev, [file.file_name]: false }));
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, userEmail, files, generatedDocuments]);

  // Combine uploaded files and generated documents
  const allFiles = [
    ...(files || []).map(f => ({ ...f, is_generated: false })),
    ...(generatedDocuments || []).map(f => ({ ...f, is_generated: true }))
  ];

  if (allFiles.length === 0 && projectFiles.length === 0) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: '#6b7280' }}>
        No files associated with this model.
      </div>
    );
  }

  return (
    <div className="multi-model-files-list">
      {/* Generated Project Section */}
      {projectFiles && projectFiles.length > 0 && (
        <div style={{ marginBottom: '1.5rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '1rem', padding: '10px' }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '0.75rem',
            padding: '0 0.5rem'
          }}>
            <h3 style={{
              fontSize: '0.9rem',
              fontWeight: '700',
              color: '#1f2937',
              margin: 0,
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              Generated Project
            </h3>
            {/* Deploy button removed */}
          </div>

          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
            backgroundColor: '#ffffff',
            borderRadius: '6px',
            border: '1px solid #e5e7eb',
            padding: '8px',
            marginTop: '8px'
          }}>
            {(() => {
              const fileTree = buildFileTree(projectFiles);
              return Object.values(fileTree)
                .sort((a, b) => {
                  if (a.type === b.type) return a.name.localeCompare(b.name);
                  return a.type === 'folder' ? -1 : 1;
                })
                .map((node) => (
                  <FileTreeItem
                    key={node.path}
                    node={node}
                    onFileClick={onProjectFileClick}
                    viewingFile={viewingFile}
                  />
                ));
            })()}
          </div>
        </div>
      )}

      {/* Existing Files Section */}
      {(allFiles.length > 0) && (
        <div style={{ padding: '0 0.5rem' }}>
          <h3 style={{
            fontSize: '0.8rem',
            fontWeight: '600',
            color: '#6b7280',
            margin: '0 0 0.75rem 0',
            textTransform: 'uppercase',
            letterSpacing: '0.05em'
          }}>
            Source Files
          </h3>
          <Collapse
            accordion
            activeKey={activeKeys}
            onChange={handlePanelChange}
            expandIconPosition="end"
            ghost
          >
            {allFiles.map((file, idx) => (
              <Panel
                header={
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, width: '100%' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 0 }}>
                      <div style={{
                        fontSize: '1.25rem',
                        lineHeight: 1,
                        flexShrink: 0
                      }}>
                        {file.file_type === 'tabular' ? '📊' : file.file_type === 'document' ? '📄' : '🖼️'}
                      </div>
                      <AntdTooltip title={file.file_name} placement="top">
                        <div style={{
                          fontWeight: 600,
                          fontSize: '0.9rem',
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          flex: 1,
                          maxWidth: '200px',
                          cursor: 'default'
                        }}>
                          {file.file_name}
                        </div>
                      </AntdTooltip>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDownload(file);
                      }}
                      disabled={downloadingFiles[file.file_name]}
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: downloadingFiles[file.file_name] ? 'not-allowed' : 'pointer',
                        padding: '0.25rem 0.5rem',
                        borderRadius: '4px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: downloadingFiles[file.file_name] ? '#9ca3af' : '#2563eb',
                        flexShrink: 0,
                        transition: 'background-color 0.2s'
                      }}
                      onMouseEnter={(e) => {
                        if (!downloadingFiles[file.file_name]) {
                          e.target.style.backgroundColor = '#f3f4f6';
                        }
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.backgroundColor = 'transparent';
                      }}
                      title="Download file"
                    >
                      {downloadingFiles[file.file_name] ? (
                        <Spinner animation="border" size="sm" />
                      ) : (
                        <FaDownload style={{ fontSize: '0.875rem' }} />
                      )}
                    </button>
                  </div>
                }
                key={String(idx)}
              >
                <div style={{ fontSize: '0.85rem', color: '#6b7280', display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
                    <Tag>{file.file_type}</Tag>
                    {file.is_generated ? (
                      <Tag color="purple">Generated</Tag>
                    ) : file.processed && (
                      <Tag color="green">Processed</Tag>
                    )}
                    <span style={{ fontSize: '0.75rem', color: '#9ca3af' }}>
                      {file.is_generated ? 'Generated' : 'Added'}: {new Date(file.created_at).toLocaleDateString()}
                    </span>
                  </div>

                  {/* File Preview */}
                  {file.is_generated && file.report_data ? (
                    <div style={{ padding: '1rem' }}>
                      <div style={{ marginBottom: '1rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '0.5rem' }}>
                        <AntdTooltip title={file.file_name} placement="top">
                          <h3 style={{
                            fontSize: '1rem',
                            fontWeight: '600',
                            margin: 0,
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            maxWidth: '100%',
                            cursor: 'default'
                          }}>
                            {file.file_name}
                          </h3>
                        </AntdTooltip>
                        <Tag color="blue" style={{ marginTop: '0.5rem' }}>Generated Report</Tag>
                      </div>
                      <div style={{ padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '4px' }}>
                        <p style={{ fontSize: '0.875rem', color: '#6b7280', margin: 0 }}>
                          This is a generated report. Click download to save as PDF.
                        </p>
                      </div>
                    </div>
                  ) : loadingFiles[file.file_name] ? (
                    <div style={{ padding: '1rem', textAlign: 'center' }}>
                      <Spinner animation="border" size="sm" />
                      <p style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: '#6b7280' }}>Loading preview...</p>
                    </div>
                  ) : fileData[file.file_name] ? (
                    <FilePreview fileData={fileData[file.file_name]} filename={file.file_name} />
                  ) : file.is_generated && file.file_preview_data ? (
                    <FilePreview fileData={file.file_preview_data} filename={file.file_name} />
                  ) : (
                    <div style={{ padding: '0.5rem', color: '#9ca3af', fontSize: '0.85rem' }}>
                      Click to expand and view preview
                    </div>
                  )}
                </div>
              </Panel>
            ))}
          </Collapse>
        </div>
      )}
    </div>
  );
};

// Component to render message content
const formatAgentErrorMessage = (raw) => {
  if (!raw) return 'An error occurred. Please try again.';
  if (/api_key|OPENAI_API_KEY/i.test(raw)) {
    return 'AI service is not configured. Set OPENAI_API_KEY in akkio-fastapi/.env and restart the backend server.';
  }
  return raw;
};

const isErrorAnswerContent = (content) => {
  try {
    const parsed = JSON.parse(content);
    const answer = parsed?.answer || '';
    return /api_key|OPENAI_API_KEY|not configured|Connection error/i.test(answer);
  } catch {
    return false;
  }
};

const MessageContent = ({ content }) => {
  try {
    const parsedContent = JSON.parse(content);
    const isError = isErrorAnswerContent(content);

    // Check if it's a multi-model answer (new format with metadata separated)
    if (parsedContent.multi_model_metadata) {
      // Merge answer with metadata for MultiModelAnswer component
      const multiModelData = {
        answer: parsedContent.answer,
        ...parsedContent.multi_model_metadata
      };
      return <MultiModelAnswer data={multiModelData} />;
    }

    // Check if it's a multi-model answer (old format - backward compatibility)
    if (parsedContent.multi_model_answer || (parsedContent.answer && parsedContent.sources && parsedContent.agents_used)) {
      return <MultiModelAnswer data={parsedContent} />;
    }

    // Check if it's a report format
    if (parsedContent.report) {
      return <ReportContent data={parsedContent} />;
    }

    // General older format answer
    if (parsedContent.answer) {
      const answerData = isError
        ? { answer: formatAgentErrorMessage(parsedContent.answer) }
        : parsedContent;
      return <GeneralAnswer data={answerData} isError={isError} />;
    }

    // New agent formats: {type, payload, explanation ? }
    if (parsedContent.type && Object.prototype.hasOwnProperty.call(parsedContent, 'payload')) {
      const { type, payload, explanation } = parsedContent;
      if (type === 'plotly') {
        if (!payload) return <AnswerText payload="No chart to display." explanation={explanation} />;
        return <AnswerPlotly figure={payload} explanation={explanation} />;
      }
      if (type === 'table') {
        const rows = Array.isArray(payload) ? payload : [];
        return <AnswerTable rows={rows} explanation={explanation} />;
      }
      if (type === 'text') {
        return <AnswerText payload={payload} explanation={explanation} />;
      }
      // Unknown type -> stringify
      return <AnswerText payload={parsedContent} />;
    }

    // Fallback to original content if structure is unknown
    return (
      <div dangerouslySetInnerHTML={{ __html: cleanHtmlContent(content) }} />
    );
  } catch (error) {
    // If parsing fails, display as HTML (original behavior)
    return (
      <div dangerouslySetInnerHTML={{ __html: cleanHtmlContent(content) }} />
    );
  }
};

const MultiAgent = ({ initialSessionId }) => {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([
    { type: 'bot', content: '{"answer": "Hello! How can I assist you today?"}' }
  ]);
  const [sessionId, setSessionId] = useState(() => initialSessionId || localStorage.getItem('multiModelSessionId') || '');
  const [activeModelName, setActiveModelName] = useState(() => localStorage.getItem('multiModelName') || '');
  const email = JSON.parse(localStorage.getItem('user'))?.email;
  const [showFilePreview, setShowFilePreview] = useState(true);
  const [multiModelFiles, setMultiModelFiles] = useState([]);
  const [loadingMultiFiles, setLoadingMultiFiles] = useState(false);
  const [showPremiumOverlay, setShowPremiumOverlay] = useState(false);
  const [generatedDocuments, setGeneratedDocuments] = useState([]);
  const [projectStructure, setProjectStructure] = useState(null);
  const [viewingProjectFile, setViewingProjectFile] = useState(null);
  const [multiModels, setMultiModels] = useState([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState('all');
  const [initialTemplateData, setInitialTemplateData] = useState(null);
  const [isShareModalOpen, setIsShareModalOpen] = useState(false);
  const [shareModel, setShareModel] = useState(null);
  const [shareInfo, setShareInfo] = useState(null);
  const [shareLoading, setShareLoading] = useState(false);
  const [publishLoading, setPublishLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);

  const handleVoiceInput = () => {
    if (isListening) {
      if (window.recognition) {
        window.recognition.stop();
      }
      setIsListening(false);
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Your browser does not support voice input. Please use Chrome or Edge.");
      return;
    }

    const recognition = new SpeechRecognition();
    window.recognition = recognition;
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setMessage(prev => (prev ? prev + ' ' + transcript : transcript));
      setIsListening(false);
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error", event.error);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.start();
  };

  // agentTemplates moved to AgentTemplates.jsx

  const getFilteredModels = () => {
    // Only return user's models, templates are handled by AgentTemplates component
    let models = multiModels;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      models = models.filter(m => (m.model_name || '').toLowerCase().includes(q));
    }
    return models;
  };
  const filteredModels = getFilteredModels();

  const handleUseTemplate = (template) => {
    setInitialTemplateData({
      model_name: template.model_name,
      system_prompt: template.system_prompt,
      temperature: 0.7, // Default for templates
      workflow: template.workflow,
      output_format: template.output_format
    });
    setIsCreateModalOpen(true);
  };

  const handleEditModel = (model) => {
    setInitialTemplateData({
      session_id: model.session_id,
      model_name: model.model_name,
      system_prompt: model.system_prompt,
      temperature: model.temperature || 0.0,
      workflow: model.workflow || model.background || '',
      output_format: model.output_format
    });
    setIsCreateModalOpen(true);
  };


  const computeShareLink = (publicId) => {
    if (!publicId || typeof window === 'undefined') return '';
    return `${window.location.origin}/chatbot/${publicId}`;
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      antdMessage.success('Copied!');
    } catch (e) {
      antdMessage.error('Copy failed');
    }
  };

  const refreshModels = async () => {
    if (!email) return;
    try {
      const response = await api.get(`/multi-model/list`);
      if (response.data?.status === 'success') {
        setMultiModels(response.data.models || []);
      }
    } catch (e) {
      // ignore
    }
  };

  const openShareEmbed = async (model, evt) => {
    if (evt?.stopPropagation) evt.stopPropagation();
    setShareModel(model);
    setIsShareModalOpen(true);
    setShareLoading(true);
    setShareInfo(null);
    try {
      const res = await api.get(`/multi-model/share-info`, {
        params: { session_id: model.session_id }
      });
      setShareInfo(res.data);
    } catch (e) {
      antdMessage.error(e?.response?.data?.detail || 'Failed to load share info');
    } finally {
      setShareLoading(false);
    }
  };

  const deployAndOpenShareEmbed = async (model, evt) => {
    if (evt?.domEvent?.stopPropagation) evt.domEvent.stopPropagation();
    if (evt?.stopPropagation) evt.stopPropagation();
    if (!model?.session_id) return;

    // Enforce readiness (backend also enforces)
    if (model.status && model.status !== 'completed') {
      antdMessage.error(`Model is not ready yet (status: ${model.status}).`);
      return;
    }

    // Open the modal immediately for UX parity with MindPal
    setShareModel(model);
    setIsShareModalOpen(true);
    setShareInfo(null);

    setPublishLoading(true);
    try {
      const res = await api.post(`/multi-model/publish`, {
        session_id: model.session_id,
        published: true
      });
      const updatedSession = res.data?.session;
      if (updatedSession) {
        setShareModel((prev) => ({ ...(prev || {}), ...updatedSession }));
      }
      setShareInfo(res.data);
      await refreshModels();
      antdMessage.success('Deployed (Published)');
    } catch (e) {
      antdMessage.error(e?.response?.data?.detail || 'Deploy failed');
    } finally {
      setPublishLoading(false);
    }
  };

  const togglePublish = async (nextPublished) => {
    if (!shareModel) return;
    setPublishLoading(true);
    try {
      const res = await api.post(`/multi-model/publish`, {
        session_id: shareModel.session_id,
        published: nextPublished
      });
      const updatedSession = res.data?.session;
      if (updatedSession) {
        setShareModel((prev) => ({ ...(prev || {}), ...updatedSession }));
        setShareInfo(res.data);
        await refreshModels();
        antdMessage.success(nextPublished ? 'Published' : 'Unpublished');
      } else {
        antdMessage.success('Updated');
      }
    } catch (e) {
      antdMessage.error(e?.response?.data?.detail || 'Failed to update publish state');
    } finally {
      setPublishLoading(false);
    }
  };

  const textareaRef = useRef(null);

  const submissionLock = useRef(false); // New lock to prevent double submission

  const handleMessageChange = (e) => {
    setMessage(e.target.value);
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && message.trim()) {
        handleSubmit(e);
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (submissionLock.current || isLoading) return; // Prevent double submit
    if (!message.trim()) return;

    submissionLock.current = true; // LOCK

    const userMessage = message;
    setMessage('');

    // Add user message and placeholder bot message for streaming
    setMessages(prev => {
      return [...prev, {
        type: 'user',
        content: userMessage,
        question: true,
        isLoading: false
      }, {
        type: 'bot',
        content: '',
        plan: '', // NEW: Plan content
        isPlanStreaming: false, // NEW
        isPlanComplete: false, // NEW
        isPlanOpen: false, // NEW: Controlled accordion state
        streaming: true,
        accumulatedAnswer: '',
        metadata: null
      }];
    });

    setIsLoading(true);

    try {
      const modelName = activeModelName || localStorage.getItem('multiModelName') || localStorage.getItem('model_name');
      const userEmail = email;

      if (!modelName) {
        antdMessage.error('Agent context not found. Please go back and select an agent.');
        setIsLoading(false);
        submissionLock.current = false;
        setMessages(prev => prev.slice(0, -2));
        return;
      }

      // Use WebSocket for multi-model query
      const socketUrl = wsUrl('/multi-model/query/ws');
      const ws = new WebSocket(socketUrl);

      ws.onopen = () => {
        const messagesToSend = [
          ...messages,
          {
            type: 'user',
            content: userMessage
          }
        ];

        ws.send(JSON.stringify(wsAuthPayload({
          model_name: modelName || '',
          query: userMessage,
          messages: messagesToSend
        })));
      };

      let accumulatedAnswer = '';
      let currentMetadata = null;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const messageType = data.type;

          setMessages(prev => {
            const botIdx = prev.length - 1; // Always refer to the last message for streaming updates

            if (messageType === 'session_id') {
              if (data.session_id && data.session_id !== sessionId) {
                setSessionId(data.session_id);
              }
            } else if (messageType === 'intent_analysis') {
              // Intent analysis received - could show this in UI if needed
              return prev;
            } else if (messageType === 'execution_plan') {
              // Execution plan received - could show this in UI if needed
              return prev;
            } else if (messageType === 'agent_start') {
              // Agent started - could show progress indicator
              return prev;
            } else if (messageType === 'agent_complete') {
              // Agent completed - could show progress indicator
              return prev;
            } else if (messageType === 'validation_progress') {
              // Validation in progress
              return prev;
            } else if (messageType === 'plan_start') {
              return prev.map((msg, idx) =>
                idx === botIdx ? {
                  ...msg,
                  isPlanStreaming: true,
                  isPlanOpen: true // Auto-open on start
                } : msg
              );
            } else if (messageType === 'plan_chunk') {
              return prev.map((msg, idx) =>
                idx === botIdx ? {
                  ...msg,
                  plan: (msg.plan || '') + (data.chunk || '') // Fix key to data.chunk
                } : msg
              );
            } else if (messageType === 'plan_complete') {
              return prev.map((msg, idx) =>
                idx === botIdx ? {
                  ...msg,
                  isPlanStreaming: false,
                  isPlanComplete: true,
                  isPlanOpen: false // Auto-close on completion
                } : msg
              );
            } else if (messageType === 'answer_start') {
              // Optional: Indicate answer starting
              return prev;
            } else if (messageType === 'answer_chunk') {
              accumulatedAnswer += data.chunk || '';
              return prev.map((msg, idx) =>
                idx === botIdx ? {
                  ...msg,
                  accumulatedAnswer: accumulatedAnswer,
                  content: JSON.stringify({ answer: accumulatedAnswer })
                } : msg
              );
            } else if (messageType === 'metadata') {
              currentMetadata = data;
              return prev.map((msg, idx) =>
                idx === botIdx ? {
                  ...msg,
                  metadata: currentMetadata
                } : msg
              );
            } else if (messageType === 'complete') {
              if (data.session_id && data.session_id !== sessionId) {
                setSessionId(data.session_id);
              }

              // Extract generated documents from metadata
              if (currentMetadata) {
                try {
                  // Check for report
                  let report = currentMetadata.report;
                  if (report && (report.heading || report.paragraphs || report.table || report.charts)) {
                    const reportTitle = report.heading || 'Generated Report';
                    const timestamp = new Date().toISOString();
                    const fileName = `${reportTitle.replace(/[^a-zA-Z0-9]/g, '_')}_${new Date().toISOString().split('T')[0]}.pdf`;

                    const generatedDoc = {
                      file_name: fileName,
                      file_type: 'document',
                      created_at: timestamp,
                      is_generated: true,
                      report_data: report,
                      generation_format: currentMetadata.generation_format
                    };

                    setGeneratedDocuments(prev => {
                      const reportHash = JSON.stringify(report);
                      const currentTime = new Date(timestamp).getTime();
                      const exists = prev.some(doc => {
                        if (doc.is_generated && doc.report_data) {
                          const existingHash = JSON.stringify(doc.report_data);
                          const docTime = new Date(doc.created_at).getTime();
                          const timeDiff = Math.abs(currentTime - docTime);
                          return existingHash === reportHash || (doc.file_name === fileName && timeDiff < 10000);
                        }
                        return false;
                      });
                      if (!exists) {
                        return [...prev, generatedDoc];
                      }
                      return prev;
                    });
                  }

                  // Check for project structure
                  const newProjectStructure = currentMetadata.project_structure;
                  if (newProjectStructure) {
                    setProjectStructure(normalizeProjectStructure(newProjectStructure));
                  }
                } catch (error) {
                  console.error('Error extracting generated documents:', error);
                }
              }

              // Finalize the message
              return prev.map((msg, idx) => {
                if (idx === botIdx) {
                  // Format final content with answer and metadata
                  const finalContent = {
                    answer: accumulatedAnswer || '',
                    multi_model_metadata: currentMetadata || {}
                  };

                  return {
                    ...msg,
                    streaming: false,
                    content: JSON.stringify(finalContent)
                  };
                }
                return msg;
              });
            } else if (messageType === 'error') {
              setIsLoading(false);
              submissionLock.current = false;
              const friendly = formatAgentErrorMessage(data.message || 'An error occurred');
              return prev.map((msg, idx) =>
                idx === botIdx ? {
                  type: 'bot',
                  content: JSON.stringify({ answer: friendly }),
                  streaming: false,
                  isPlanStreaming: false,
                  plan: msg.plan || '',
                } : msg
              );
            }
            return prev;
          });
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setMessages(prev => {
          const botIdx = prev.length - 1;
          return prev.map((msg, idx) =>
            idx === botIdx ? {
              type: 'bot',
              content: JSON.stringify({ answer: 'Connection error. Please try again.' }),
              streaming: false
            } : msg
          );
        });
        setIsLoading(false);
      };

      ws.onclose = () => {
        setIsLoading(false);
        submissionLock.current = false;
        window.dispatchEvent(new Event('usage_updated'));
      };

      return;
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => {
        const botIdx = prev.length - 1;
        return prev.map((msg, idx) =>
          idx === botIdx ? {
            type: 'bot',
            content: JSON.stringify({ answer: 'Sorry, there was an error processing your request.' }),
            streaming: false
          } : msg
        );
      });
      setIsLoading(false);
      return; // Exit after handling error

      // Note: Generated documents extraction moved to WebSocket complete handler below
      // This code block is now unreachable but kept for reference
      if (false) {
        // eslint-disable-next-line no-unused-vars
        const response = null; // Dummy variable for unreachable code reference
        try {
          const responseData = typeof response.data === 'object' ? response.data : JSON.parse(response.data);

          console.log('Response data for document extraction:', responseData);

          // Check if response contains a report (which can be downloaded as PDF)
          // Check multiple possible locations for report data
          let report = null;
          if (responseData.report) {
            report = responseData.report;
          } else if (responseData.multi_model_metadata?.report) {
            report = responseData.multi_model_metadata.report;
          } else if (responseData.answer && typeof responseData.answer === 'object' && responseData.answer.report) {
            report = responseData.answer.report;
          }

          // Check for project structure
          const newProjectStructure = responseData.project_structure || responseData.multi_model_metadata?.project_structure;
          if (newProjectStructure) {
            setProjectStructure(normalizeProjectStructure(newProjectStructure));
          }

          // If no report but generation_format indicates PDF/document creation and answer exists, convert answer to report
          const generationFormat = responseData.generation_format || responseData.multi_model_metadata?.generation_format;
          const shouldCreateReport = (generationFormat && generationFormat !== 'null' && (generationFormat.toLowerCase().includes('pdf') || generationFormat.toLowerCase().includes('doc'))) ||
            (responseData.answer && typeof responseData.answer === 'string' && (responseData.answer.includes('<h3>') || responseData.answer.includes('create') || responseData.answer.includes('generate')));

          if (!report && shouldCreateReport && responseData.answer) {
            const answerHtml = typeof responseData.answer === 'string' ? responseData.answer : '';

            // Clean the HTML content first
            const cleanedAnswerHtml = cleanHtmlContent(answerHtml);

            // Extract heading from first h3 tag
            const headingMatch = cleanedAnswerHtml.match(/<h3[^>]*>(.*?)<\/h3>/i);
            let heading = headingMatch ? headingMatch[1].replace(/<[^>]*>/g, '').trim() : 'Generated Document';

            // If no h3, try to extract from first meaningful text
            if (heading === 'Generated Document' || !heading) {
              const firstTextMatch = cleanedAnswerHtml.match(/<p[^>]*>(.*?)<\/p>/i) || cleanedAnswerHtml.match(/([^<>\n]{20,})/);
              if (firstTextMatch) {
                heading = firstTextMatch[1].replace(/<[^>]*>/g, '').substring(0, 50).trim();
              }
            }

            // Split answer into paragraphs
            // Replace h3 with h4 for better formatting in report
            let processedHtml = cleanedAnswerHtml.replace(/<h3[^>]*>/gi, '<h4>').replace(/<\/h3>/gi, '</h4>');

            const paragraphs = [];

            // Split by block-level elements (p, h4, ul, ol, div)
            // First, extract all block elements
            const blockElements = processedHtml.match(/<(p|h4|ul|ol|div)[^>]*>[\s\S]*?<\/\1>/gi) ||
              processedHtml.match(/<(p|h4|ul|ol|div)[^>]*\/>/gi) ||
              [];

            if (blockElements.length > 0) {
              blockElements.forEach(element => {
                // Clean up the element
                let cleanElement = element.trim();
                // Ensure it's wrapped properly
                if (!cleanElement.match(/^<p/i) && !cleanElement.match(/^<h4/i) && !cleanElement.match(/^<ul/i) && !cleanElement.match(/^<ol/i)) {
                  cleanElement = `<p>${cleanElement}</p>`;
                }
                paragraphs.push(cleanElement);
              });
            } else {
              // If no block elements found, split by line breaks or use whole content
              const lines = processedHtml.split(/\n+/).filter(line => line.trim().length > 0);
              if (lines.length > 0) {
                lines.forEach(line => {
                  const trimmed = line.trim();
                  if (trimmed) {
                    // If it doesn't start with a tag, wrap it in p
                    if (!trimmed.match(/^</)) {
                      paragraphs.push(`<p>${trimmed}</p>`);
                    } else {
                      paragraphs.push(trimmed);
                    }
                  }
                });
              } else {
                // Fallback: use the whole HTML as one paragraph
                paragraphs.push(`<p>${processedHtml}</p>`);
              }
            }

            // If still no paragraphs, use the whole HTML
            if (paragraphs.length === 0) {
              paragraphs.push(processedHtml);
            }

            // Create report object from answer
            report = {
              heading: heading || 'Generated Document',
              paragraphs: paragraphs,
              table: null,
              charts: []
            };

            console.log('Converted answer to report:', report);
          }

          if (report && (report.heading || report.paragraphs || report.table || report.charts)) {
            const reportTitle = report.heading || 'Generated Report';
            const timestamp = new Date().toISOString();
            const fileName = `${reportTitle.replace(/[^a-zA-Z0-9]/g, '_')}_${new Date().toISOString().split('T')[0]}.pdf`;

            // Create a generated document entry for the report
            const generatedDoc = {
              file_name: fileName,
              file_type: 'document',
              created_at: timestamp,
              is_generated: true,
              report_data: report,
              generation_format: responseData.generation_format || responseData.multi_model_metadata?.generation_format
            };

            console.log('Adding generated document:', generatedDoc);

            setGeneratedDocuments(prev => {
              // Avoid duplicates by checking filename and report content hash
              const reportHash = JSON.stringify(report);
              const currentTime = new Date(timestamp).getTime();
              const exists = prev.some(doc => {
                if (doc.is_generated && doc.report_data) {
                  const existingHash = JSON.stringify(doc.report_data);
                  // Check if same content (by hash) or same filename within last 10 seconds
                  const docTime = new Date(doc.created_at).getTime();
                  const timeDiff = Math.abs(currentTime - docTime);
                  return existingHash === reportHash || (doc.file_name === fileName && timeDiff < 10000);
                }
                // Also check by filename alone
                const docTime = new Date(doc.created_at).getTime();
                const timeDiff = Math.abs(currentTime - docTime);
                return doc.file_name === fileName && timeDiff < 10000;
              });
              if (!exists) {
                console.log('Adding new generated document to list:', generatedDoc);
                return [...prev, generatedDoc];
              }
              console.log('Document already exists (duplicate detected), skipping');
              return prev;
            });
          }

          // Check for other generated documents in the response
          if (responseData.generated_documents && Array.isArray(responseData.generated_documents)) {
            responseData.generated_documents.forEach(doc => {
              const generatedDoc = {
                file_name: doc.file_name || `generated_${Date.now()}.${doc.file_type || 'pdf'}`,
                file_type: doc.file_type || 'document',
                created_at: doc.created_at || new Date().toISOString(),
                is_generated: true,
                file_data: doc.file_data || doc.url,
                file_preview_data: doc
              };

              setGeneratedDocuments(prev => {
                const exists = prev.some(d => d.file_name === generatedDoc.file_name);
                if (!exists) {
                  console.log('Adding new generated document to list:', generatedDoc);
                  return [...prev, generatedDoc];
                }
                return prev;
              });
            });
          }
        } catch (error) {
          // If parsing fails, continue without extracting generated documents
          console.error('Could not extract generated documents from response:', error);
        }
      } // End of try-catch
    } // End of unreachable block

    // Handle generated documents extraction from WebSocket complete message
    // This is done in the ws.onmessage handler above when 'complete' is received
  };

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Reset textarea height when message is cleared
  useEffect(() => {
    if (textareaRef.current && !message) {
      textareaRef.current.style.height = 'auto';
    }
  }, [message]);



  // Track processed message indices to avoid reprocessing
  const processedMessageIndicesRef = useRef(new Set());

  // Helper to normalize project structure (handle nested objects vs flat list)
  const normalizeProjectStructure = (structure) => {
    if (!structure) return null;

    // If it already has the desired format
    if (structure.files && Array.isArray(structure.files)) {
      return structure;
    }

    // Otherwise, recursively flatten
    const files = [];

    const traverse = (obj, currentPath = '') => {
      Object.entries(obj).forEach(([key, value]) => {
        const path = currentPath ? `${currentPath}/${key}` : key;
        if (typeof value === 'string') {
          files.push({ path, content: value });
        } else if (typeof value === 'object' && value !== null) {
          traverse(value, path);
        }
      });
    };

    traverse(structure);

    // Heuristic: If we extracted files but no title, use the root key or generic name
    return {
      project_title: structure.project_title || Object.keys(structure)[0] || 'Generated Project',
      files: files,
      deployment_config: structure.deployment_config || null
    };
  };

  // Reset processed indices when session changes
  useEffect(() => {
    processedMessageIndicesRef.current.clear();
    setGeneratedDocuments([]); // Clear documents when session changes
    setProjectStructure(null); // Clear project structure on session change
    setViewingProjectFile(null); // Clear viewing file on session change
  }, [sessionId, initialSessionId]);

  // Load generated documents from messages history
  useEffect(() => {
    const loadGeneratedDocumentsFromMessages = () => {
      const newDocs = [];

      messages.forEach((msg, index) => {
        // Skip if we've already processed this message
        if (processedMessageIndicesRef.current.has(index)) {
          return;
        }

        if (msg.type === 'bot' && msg.content) {
          try {
            const parsedContent = typeof msg.content === 'string' ? JSON.parse(msg.content) : msg.content;

            // Check for report in various locations
            let report = null;
            if (parsedContent.report) {
              report = parsedContent.report;
            } else if (parsedContent.multi_model_metadata?.report) {
              report = parsedContent.multi_model_metadata.report;
            } else if (parsedContent.answer && typeof parsedContent.answer === 'object' && parsedContent.answer.report) {
              report = parsedContent.answer.report;
            }

            if (report && (report.heading || report.paragraphs || report.table || report.charts)) {
              const reportTitle = report.heading || 'Generated Report';
              const timestamp = new Date().toISOString();
              const fileName = `${reportTitle.replace(/[^a-zA-Z0-9]/g, '_')}_${index}_${Date.now()}.pdf`;

              const generatedDoc = {
                file_name: fileName,
                file_type: 'document',
                created_at: timestamp,
                is_generated: true,
                report_data: report,
                generation_format: parsedContent.generation_format || parsedContent.multi_model_metadata?.generation_format,
                message_index: index
              };

              newDocs.push(generatedDoc);
              processedMessageIndicesRef.current.add(index);
            }

            // Check for other generated documents
            if (parsedContent.generated_documents && Array.isArray(parsedContent.generated_documents)) {
              parsedContent.generated_documents.forEach(doc => {
                const generatedDoc = {
                  file_name: doc.file_name || `generated_${index}_${Date.now()}.${doc.file_type || 'pdf'}`,
                  file_type: doc.file_type || 'document',
                  created_at: doc.created_at || new Date().toISOString(),
                  is_generated: true,
                  file_data: doc.file_data || doc.url,
                  file_preview_data: doc,
                  message_index: index
                };

                newDocs.push(generatedDoc);
              });
              processedMessageIndicesRef.current.add(index);
            }

            // Extract project structure from history if not already set
            const historyProjectStructure = parsedContent.project_structure || parsedContent.multi_model_metadata?.project_structure;
            if (historyProjectStructure && !projectStructure) {
              setProjectStructure(normalizeProjectStructure(historyProjectStructure));
            }

          } catch (error) {
            // Skip messages that can't be parsed
            console.log('Could not parse message for document extraction:', error);
          }
        }
      });

      // Update state with new documents found
      if (newDocs.length > 0) {
        console.log('Adding new generated documents from messages:', newDocs);
        setGeneratedDocuments(prev => {
          // Merge with existing, avoiding duplicates by message_index
          const merged = [...prev];
          newDocs.forEach(newDoc => {
            const exists = merged.some(existing =>
              existing.message_index === newDoc.message_index &&
              existing.is_generated
            );
            if (!exists) {
              merged.push(newDoc);
            }
          });
          return merged;
        });
      }
    };

    // Load documents when messages change
    if (messages.length > 0) {
      loadGeneratedDocumentsFromMessages();
    }
  }, [messages, projectStructure]); // Run when messages array changes

  // Fetch Multi-Model Files - this one is relevant for this page
  // The fetchFileData for single files has been removed.

  // Fetch Multi-Model Files
  useEffect(() => {
    const fetchMultiModelFiles = async () => {
      const mSessionId = sessionId || localStorage.getItem('multiModelSessionId');

      if (mSessionId) {
        setLoadingMultiFiles(true);
        try {
          const response = await api.get(`/multi-model/files`, {
            params: {
              session_id: mSessionId,
            }
          });
          if (response.data.status === 'success') {
            setMultiModelFiles(response.data.files || []);
          }
        } catch (error) {
          console.error('Error fetching multi-model files:', error);
        } finally {
          setLoadingMultiFiles(false);
        }
      } else {
        setMultiModelFiles([]);
      }
    };

    if (sessionId) {
      fetchMultiModelFiles();
    }
  }, [sessionId, email]);

  // Fetch session details on load if sessionId exists (to get model name for welcome message)
  useEffect(() => {
    const fetchSessionDetails = async () => {
      if (!sessionId || !email) return;

      try {
        // Use progress endpoint to get model name
        const response = await api.get(`/multi-model/progress`, {
          params: { session_id: sessionId }
        });

        if (response.data && response.data.model_name) {
          const mName = response.data.model_name;
          localStorage.setItem('multiModelName', mName);
          setActiveModelName(mName);

          // Update the welcome message if it's the only message and is generic
          setMessages(prev => {
            if (prev.length === 1 && prev[0].type === 'bot') {
              const contentStr = prev[0].content;
              if (contentStr.includes("Hello! How can I assist you today?")) {
                return [{
                  type: 'bot',
                  content: JSON.stringify({ answer: `Hi! I'm ${mName}. How can I help you today?` })
                }];
              }
            }
            return prev;
          });
        }
      } catch (e) {
        if (e.response?.status === 404) {
          setSessionId('');
          setActiveModelName('');
          localStorage.removeItem('multiModelSessionId');
          localStorage.removeItem('multiModelName');
          antdMessage.warning('Session expired or was deleted.');
        } else {
          console.error("Failed to fetch session details:", e);
        }
      }
    };

    fetchSessionDetails();
  }, [sessionId, email]);

  // Fetch Multi-Models List
  useEffect(() => {
    const fetchMultiModels = async () => {
      if (!email) return;

      setLoadingModels(true);
      try {
        const response = await api.get(`/multi-model/list`);
        if (response.data.status === 'success') {
          setMultiModels(response.data.models || []);
        }
      } catch (error) {
        console.error('Error fetching multi-models:', error);
        antdMessage.error('Failed to load multi-models');
      } finally {
        setLoadingModels(false);
      }
    };

    // Only fetch if no sessionId is selected (showing list view)
    if (!sessionId && !initialSessionId) {
      fetchMultiModels();
    }
  }, [email, sessionId, initialSessionId]);

  const handleModelSelect = (model) => {
    setSessionId(model.session_id);
    setActiveModelName(model.model_name);
    localStorage.setItem('multiModelSessionId', model.session_id);
    localStorage.setItem('multiModelName', model.model_name);
    localStorage.setItem('selectedFileType', 'multi-model');
    setGeneratedDocuments([]);
    setProjectStructure(null);
    setViewingProjectFile(null);
    setMessages([
      { type: 'bot', content: JSON.stringify({ answer: `Hi! I'm ${model.model_name}. How can I help you today?` }) }
    ]);
  };

  const handleBackToList = () => {
    setSessionId('');
    setActiveModelName('');
    localStorage.removeItem('multiModelSessionId');
    localStorage.removeItem('multiModelName');
    setGeneratedDocuments([]);
    setProjectStructure(null);
    setViewingProjectFile(null);
    setMultiModelFiles([]);
    setMessages([
      { type: 'bot', content: '{"answer": "Hello! How can I assist you today?"}' }
    ]);
    refreshModels();
  };

  const handleDeleteModel = async (model, e) => {
    e.stopPropagation(); // Prevent card click

    // Show confirmation dialog
    if (window.confirm(`Are you sure you want to delete "${model.model_name}"? This action cannot be undone.`)) {
      try {
        const response = await api.delete(`/multi-model/delete`, {
          params: {
            session_id: model.session_id,
          }
        });

        if (response.data.status === 'success') {
          antdMessage.success(response.data.message || 'Model deleted successfully');

          // Refresh the models list
          if (email) {
            const listResponse = await api.get(`/multi-model/list`, {
              params: {}
            });
            if (listResponse.data.status === 'success') {
              setMultiModels(listResponse.data.models || []);
            }
          }

          // If the deleted model was the current session, reset
          if (sessionId === model.session_id) {
            setSessionId('');
            setActiveModelName('');
            localStorage.removeItem('multiModelSessionId');
            localStorage.removeItem('multiModelName');
            setMessages([
              { type: 'bot', content: '{"answer": "Hello! How can I assist you today?"}' }
            ]);

          }
        }
      } catch (error) {
        console.error('Error deleting model:', error);
        antdMessage.error(error.response?.data?.detail || 'Failed to delete model. Please try again.');
      }
    }
  };

  const handleTrainingComplete = (result) => {
    if (result?.session_id) {
      const name = result.model_name || result.session?.model_name || 'Agent';
      setSessionId(result.session_id);
      setActiveModelName(name);
      localStorage.setItem('multiModelSessionId', result.session_id);
      localStorage.setItem('multiModelName', name);
      localStorage.setItem('selectedFileType', 'multi-model');
      setGeneratedDocuments([]);
      setProjectStructure(null);
      setMessages([
        { type: 'bot', content: JSON.stringify({ answer: `Hi! I'm ${name}. How can I help you today?` }) }
      ]);
      antdMessage.success('Agent created! You can start chatting now.');
    } else {
      antdMessage.success('Model saved successfully!');
    }

    refreshModels();
  };

  // Show list view if no sessionId is selected
  if (!sessionId && !initialSessionId) {
    return (
      <div style={{ minHeight: 'calc(100vh - 120px)', backgroundColor: '#fff', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' }}>

        {/* Header Tabs */}
        <div style={{ marginBottom: '24px' }}>
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={[
              { key: 'all', label: 'All' },
              { key: 'templates', label: 'Templates' }
            ]}
            tabBarStyle={{ borderBottom: 'none' }}
          />
        </div>

        {activeTab === 'all' && (
          <div style={{ marginBottom: '8px', fontWeight: '500', color: '#111827' }}>All</div>
        )}

        {/* Search Bar */}
        <div style={{ marginBottom: '24px' }}>
          <Input
            prefix={<FaSearch style={{ color: '#9ca3af' }} />}
            placeholder="Search..."
            size="large"
            style={{
              borderRadius: '8px',
              maxWidth: '100%',
              backgroundColor: '#f3f4f6',
              border: 'none',
              padding: '8px 12px'
            }}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>



        {/* Agents Section */}
        <div>
          <h3 style={{ fontSize: '14px', color: '#6b7280', fontWeight: '500', marginBottom: '12px' }}>Agents</h3>


          {activeTab === 'templates' ? (
            <AgentTemplates searchQuery={searchQuery} onUseTemplate={handleUseTemplate} />
          ) : loadingModels ? (
            <div style={{ padding: '40px', textAlign: 'center' }}><Spinner animation="border" /></div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px' }}>
              {/* New Agent Card - Only show in 'all' tab */}
              {activeTab === 'all' && (
                <div
                  onClick={() => setIsCreateModalOpen(true)}
                  style={{
                    backgroundColor: '#f3f4f6',
                    borderRadius: '12px',
                    border: 'none',
                    minHeight: '200px',
                    height: 'auto',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#e5e7eb'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#f3f4f6'; }}
                >
                  <div style={{ fontSize: '24px', color: '#3b82f6', marginBottom: '8px' }}><FaPlus /></div>
                  <div style={{ color: '#3b82f6', fontWeight: '500' }}>New agent</div>
                </div>
              )}

              {/* Filtered Models (Agents Only) */}
              {filteredModels.map(model => (
                <div
                  key={model.session_id || model.id}
                  style={{
                    backgroundColor: '#ffffff',
                    borderRadius: '12px',
                    border: '1px solid #e5e7eb',
                    padding: '20px',
                    display: 'flex',
                    flexDirection: 'column',
                    minHeight: '200px',
                    height: 'auto',
                    transition: 'all 0.2s',
                    boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                    position: 'relative'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)';
                    e.currentTarget.style.borderColor = '#d1d5db';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.boxShadow = '0 1px 2px rgba(0,0,0,0.05)';
                    e.currentTarget.style.borderColor = '#e5e7eb';
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                    <div style={{
                      width: '40px',
                      height: '40px',
                      borderRadius: '50%',
                      backgroundColor: '#f3f4f6',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#4b5563'
                    }}>
                      <FaBrain size={20} />
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      {model.published && (
                        <Tag color="green" style={{ margin: 0, borderRadius: 999, fontWeight: 600 }}>
                          Published
                        </Tag>
                      )}
                      {/* Menu for non-templates */}
                      {!model.isTemplate && (
                        <Dropdown
                          overlay={
                            <Menu>
                              <Menu.Item key="edit" onClick={(e) => { e.domEvent.stopPropagation(); handleEditModel(model); }} icon={<FaCheckCircle />}>
                                Edit
                              </Menu.Item>
                              {model.published ? (
                                <Menu.Item key="share_embed" onClick={(e) => { e.domEvent.stopPropagation(); openShareEmbed(model); }} icon={<FaCheckCircle />}>
                                  Share & Embed
                                </Menu.Item>
                              ) : (
                                <Menu.Item key="deploy" onClick={(e) => deployAndOpenShareEmbed(model, e)} icon={<FaCheckCircle />}>
                                  Deploy
                                </Menu.Item>
                              )}
                              <Menu.Item key="delete" onClick={(e) => handleDeleteModel(model, e.domEvent)} danger icon={<FaTrashAlt />}>
                                Delete
                              </Menu.Item>
                            </Menu>
                          }
                          trigger={['click']}
                        >
                          <Button type="text" shape="circle" icon={<FaEllipsisV style={{ color: '#9ca3af' }} />} onClick={(e) => e.stopPropagation()} />
                        </Dropdown>
                      )}
                    </div>
                  </div>

                  <div style={{ marginBottom: '8px' }}>
                    <h3 style={{ fontSize: '16px', fontWeight: '600', color: '#111827', margin: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {model.model_name}
                    </h3>
                  </div>

                  <div style={{
                    fontSize: '14px',
                    color: '#6b7280',
                    flex: 1,
                    overflow: 'hidden',
                    display: '-webkit-box',
                    WebkitLineClamp: 3,
                    WebkitBoxOrient: 'vertical',
                    marginBottom: '16px',
                    lineHeight: '1.5'
                  }}>
                    {model.system_prompt || "No description provided."}
                  </div>

                  <div>
                    <Button
                      type="primary"
                      block
                      style={{ borderRadius: '6px', background: '#eef2ff', color: '#4f46e5', border: 'none', fontWeight: '500' }}
                      onClick={() => handleModelSelect(model)}
                    >
                      Start chat
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}

        </div>

        {/* Create Modal */}
        <Modal
          open={isCreateModalOpen}
          onCancel={() => {
            setIsCreateModalOpen(false);
            setInitialTemplateData(null);
          }}
          footer={null}
          width={800}
          destroyOnClose
          centered
          bodyStyle={{ padding: 0 }}
          closeIcon={null}
          style={{ padding: 0 }}
        >
          <div style={{ position: 'relative' }}>
            <Button
              type="text"
              icon={<span>✕</span>}
              onClick={() => {
                setIsCreateModalOpen(false);
                setInitialTemplateData(null);
              }}
              style={{ position: 'absolute', right: 10, top: 10, zIndex: 10, fontSize: '16px' }}
            />
            <MultiModelTraining
              initialData={initialTemplateData}
              onTrainingComplete={(result) => {
                handleTrainingComplete(result);
                setIsCreateModalOpen(false);
                setInitialTemplateData(null);
              }}
            />
          </div>
        </Modal>

        {/* Share & Embed Modal (MindPal-style) */}
        <Modal
          open={isShareModalOpen}
          onCancel={() => {
            setIsShareModalOpen(false);
            setShareModel(null);
            setShareInfo(null);
          }}
          footer={null}
          width={820}
          destroyOnClose
          centered
          bodyStyle={{ padding: 0 }}
        >
          <div style={{ padding: 24, position: 'relative' }}>
            <Button
              type="text"
              icon={<span>✕</span>}
              onClick={() => {
                setIsShareModalOpen(false);
                setShareModel(null);
                setShareInfo(null);
              }}
              style={{ position: 'absolute', right: 12, top: 12, zIndex: 10, fontSize: '16px' }}
            />

            <div style={{ fontSize: 14, color: '#6b7280', fontWeight: 600, marginBottom: 6 }}>Share & Embed</div>
            <div style={{ fontSize: 22, fontWeight: 800, color: '#111827', marginBottom: 18 }}>
              {shareModel?.model_name || 'Agent'}
            </div>

            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              padding: 14,
              border: '1px solid #e5e7eb',
              borderRadius: 12,
              marginBottom: 18
            }}>
              <div style={{ flex: 1, color: '#111827', fontWeight: 600 }}>
                Published: <span style={{ fontWeight: 500, color: '#6b7280' }}>Your chatbot is live and accessible to anyone with the link</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                {publishLoading && <Spinner animation="border" size="sm" />}
                <Switch
                  checked={!!shareModel?.published}
                  onChange={(checked) => togglePublish(checked)}
                  disabled={publishLoading}
                  style={{ background: shareModel?.published ? '#4f46e5' : undefined }}
                />
              </div>
            </div>

            <Tabs
              defaultActiveKey="share"
              items={[
                {
                  key: 'share',
                  label: 'Share Link',
                  children: (
                    <div style={{ paddingTop: 12 }}>
                      {shareLoading ? (
                        <div style={{ padding: 28, textAlign: 'center' }}><Spinner animation="border" size="sm" /></div>
                      ) : (
                        <>
                          <div style={{ color: '#6b7280', marginBottom: 10 }}>
                            Anyone with this link can chat with your chatbot:
                          </div>
                          <div style={{
                            display: 'flex',
                            gap: 12,
                            alignItems: 'center'
                          }}>
                            <Input
                              value={
                                (shareInfo?.share_url) ||
                                computeShareLink(shareModel?.public_id) ||
                                ''
                              }
                              readOnly
                              placeholder="Publish to generate a share link"
                              size="large"
                              style={{ borderRadius: 10 }}
                            />
                            <Button
                              size="large"
                              disabled={!shareModel?.published || !(shareInfo?.share_url || shareModel?.public_id)}
                              onClick={() => copyToClipboard((shareInfo?.share_url) || computeShareLink(shareModel?.public_id))}
                            >
                              Copy
                            </Button>
                          </div>
                        </>
                      )}
                    </div>
                  )
                },
                {
                  key: 'embed',
                  label: 'Embed Code',
                  children: (
                    <div style={{ paddingTop: 12 }}>
                      {shareLoading ? (
                        <div style={{ padding: 28, textAlign: 'center' }}><Spinner animation="border" size="sm" /></div>
                      ) : (
                        <>
                          <div style={{ color: '#6b7280', marginBottom: 10 }}>
                            Copy and paste this code anywhere on your website:
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
                            <Button
                              disabled={!shareModel?.published || !shareInfo?.embed_code}
                              onClick={() => copyToClipboard(shareInfo?.embed_code || '')}
                            >
                              Copy
                            </Button>
                          </div>
                          <Input.TextArea
                            value={shareInfo?.embed_code || ''}
                            readOnly
                            rows={6}
                            placeholder="Publish to generate embed code"
                            style={{ fontFamily: 'monospace', borderRadius: 10 }}
                          />
                        </>
                      )}
                    </div>
                  )
                },
                {
                  key: 'bubble',
                  label: 'Chat Bubble',
                  children: (
                    <div style={{ paddingTop: 12 }}>
                      {shareLoading ? (
                        <div style={{ padding: 28, textAlign: 'center' }}><Spinner animation="border" size="sm" /></div>
                      ) : (
                        <>
                          <div style={{ color: '#6b7280', marginBottom: 10 }}>
                            Add a floating chat button on your website:
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 8 }}>
                            <Button
                              disabled={!shareModel?.published || !shareInfo?.chat_bubble_code}
                              onClick={() => copyToClipboard(shareInfo?.chat_bubble_code || '')}
                            >
                              Copy
                            </Button>
                          </div>
                          <Input.TextArea
                            value={shareInfo?.chat_bubble_code || ''}
                            readOnly
                            rows={10}
                            placeholder="Publish to generate chat bubble code"
                            style={{ fontFamily: 'monospace', borderRadius: 10 }}
                          />
                        </>
                      )}
                    </div>
                  )
                }
              ]}
            />
          </div>
        </Modal>

      </div>
    );
  }

  return (
    <div className="multi-agent-chat">
      <div className="multi-agent-chat__header">
        <Button type="text" onClick={handleBackToList} style={{ padding: '4px 8px', fontWeight: 500 }}>
          ← Back to agents
        </Button>
        <span className="multi-agent-chat__title">{activeModelName || 'Agent'}</span>
      </div>

      <div className="multi-agent-chat__body">
      {/* Main Chat Container - conditionally render chat or code viewer */}
      <div className="chat-container" style={{ flex: 1, minWidth: 0 }}>
        {viewingProjectFile ? (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column', backgroundColor: '#1e1e1e', color: '#d4d4d4' }}>
            <div style={{
              padding: '10px 20px',
              backgroundColor: '#252526',
              borderBottom: '1px solid #333',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span style={{ fontSize: '14px', fontFamily: 'monospace' }}>{viewingProjectFile.path}</span>
              <button
                onClick={() => setViewingProjectFile(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#d4d4d4',
                  cursor: 'pointer',
                  fontSize: '16px'
                }}
              >
                ✕ Close
              </button>
            </div>
            <div style={{ flex: 1, overflow: 'auto', padding: '20px' }}>
              <pre style={{ margin: 0, fontFamily: 'Consolas, "Courier New", monospace', fontSize: '14px', lineHeight: '1.5' }}>
                {viewingProjectFile.content}
              </pre>
            </div>
          </div>
        ) : (
          <div className="chat-window">
            <div className="chat-messages">
              {messages.map((msg, index) => {
                const isUser = msg.type === 'user';
                const isBot = msg.type === 'bot';
                const isErrorBubble = isBot && !msg.streaming && isErrorAnswerContent(msg.content || '');

                return (
                  <div
                    key={index}
                    className={`chat-message-row ${isUser ? 'chat-message-row--user' : 'chat-message-row--bot'}`}
                  >
                    <div
                      className={`chat-message-bubble ${
                        isUser
                          ? 'chat-message-bubble--user'
                          : isErrorBubble
                            ? 'chat-message-bubble--error'
                            : 'chat-message-bubble--bot'
                      }`}
                    >
                      {(msg?.content || msg?.streaming || msg?.plan || msg?.isPlanStreaming) && (
                        <>
                          {isBot ? (
                            <div className="bot-message-content">
                              {(msg.plan || msg.isPlanStreaming) && (
                                <details
                                  open={msg.isPlanOpen}
                                  className="plan-details"
                                >
                                  <summary
                                    onClick={(e) => {
                                      e.preventDefault();
                                      setMessages(current => current.map((m, i) =>
                                        i === index ? { ...m, isPlanOpen: !m.isPlanOpen } : m
                                      ));
                                    }}
                                  >
                                    <span>Planning Step</span>
                                    {msg.isPlanStreaming && (
                                      <span className="plan-details__status">(Thinking...)</span>
                                    )}
                                  </summary>
                                  <div className="plan-details__body">
                                    <PlanContent planText={msg.plan} />
                                  </div>
                                </details>
                              )}

                              {msg.streaming ? (
                                msg.accumulatedAnswer ? (
                                  <AnswerBody answer={msg.accumulatedAnswer} isStreaming />
                                ) : (
                                  !msg.isPlanStreaming && !msg.plan ? (
                                    <div className="thinking-indicator">Thinking...</div>
                                  ) : null
                                )
                              ) : (
                                <MessageContent content={msg.content || '{"answer": ""}'} />
                              )}
                            </div>
                          ) : (
                            <div className="user-message-text">{msg.content}</div>
                          )}
                        </>
                      )}
                    </div>
                    {msg.isLoading && (
                      <div className="spinner-container">
                        <Spinner animation="border" role="status" size="sm">
                          <span className="visually-hidden">Loading...</span>
                        </Spinner>
                      </div>
                    )}
                  </div>
                );
              })}
              <div ref={messagesEndRef} />
            </div>

            <form onSubmit={handleSubmit} className="chat-input-form" style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-end', padding: '1rem', borderTop: '1px solid #e5e7eb', backgroundColor: '#ffffff' }}>
              <div className="input-container3" style={{ flex: 1, position: 'relative', display: 'flex', alignItems: 'flex-end' }}>
                <textarea
                  ref={textareaRef}
                  className="chat-input"
                  value={message}
                  onChange={handleMessageChange}
                  onKeyDown={handleKeyDown}
                  placeholder={isListening ? "Listening..." : "Ask something..."}
                  rows={1}
                  style={{
                    width: '100%',
                    minHeight: '44px',
                    maxHeight: '200px',
                    padding: '12px 40px 12px 16px',
                    border: '1px solid #d1d5db',
                    borderRadius: '12px',
                    fontSize: '15px',
                    fontFamily: 'inherit',
                    resize: 'none',
                    overflowY: 'auto',
                    outline: 'none',
                    lineHeight: '1.5',
                    backgroundColor: '#ffffff',
                    transition: 'border-color 0.2s',
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = '#2563eb';
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = '#d1d5db';
                  }}
                  disabled={isLoading}
                />
                <div
                  onClick={handleVoiceInput}
                  style={{
                    position: 'absolute',
                    right: '10px',
                    bottom: '12px',
                    cursor: 'pointer',
                    color: isListening ? '#ef4444' : '#6b7280',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '5px',
                    borderRadius: '50%',
                    backgroundColor: isListening ? '#fee2e2' : 'transparent',
                    transition: 'all 0.2s',
                    zIndex: 10
                  }}
                  title="Voice Input"
                >
                  <FaMicrophone size={16} />
                </div>
              </div>
              <button
                type="submit"
                className="send-button"
                disabled={isLoading || !message.trim()}
                style={{
                  padding: '12px 24px',
                  backgroundColor: isLoading || !message.trim() ? '#9ca3af' : '#2563eb',
                  color: 'white',
                  border: 'none',
                  borderRadius: '12px',
                  cursor: isLoading || !message.trim() ? 'not-allowed' : 'pointer',
                  fontSize: '15px',
                  fontWeight: '500',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minWidth: '80px',
                  transition: 'background-color 0.2s',
                  height: '44px'
                }}
                onMouseEnter={(e) => {
                  if (!isLoading && message.trim()) {
                    e.target.style.backgroundColor = '#1d4ed8';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isLoading && message.trim()) {
                    e.target.style.backgroundColor = '#2563eb';
                  }
                }}
              >
                {isLoading ? (
                  <Spinner
                    animation="border"
                    size="sm"
                    role="status"
                    aria-hidden="true"
                  />
                ) : (
                  'Send'
                )}
              </button>
            </form>
          </div>
        )}
      </div>

      {/* File Preview Panel - Right Side */}
      {showFilePreview && (
        <div style={{
          width: '400px',
          minWidth: '300px',
          backgroundColor: '#ffffff',
          border: '1px solid #e5e7eb',
          borderRadius: '8px',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
        }}>
          <div style={{
            padding: '1rem',
            borderBottom: '1px solid #e5e7eb',
            backgroundColor: '#f9fafb',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <h2 style={{
              fontSize: '1rem',
              fontWeight: '600',
              margin: 0,
              color: '#1f2937'
            }}>
              {(loadingMultiFiles || multiModelFiles.length > 0)
                ? 'Multi Agent Sources'
                : 'File Preview'}
            </h2>
            <button
              onClick={() => setShowFilePreview(false)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                fontSize: '1.25rem',
                color: '#6b7280',
                padding: '0.25rem 0.5rem',
                borderRadius: '4px'
              }}
              title="Hide preview"
            >
              ×
            </button>
          </div>
          <div style={{
            flex: 1,
            overflow: 'auto',
            backgroundColor: '#ffffff'
          }}>
            {/* Always show Multi-Model Files List for this page */}
            {loadingMultiFiles ? (
              <div style={{ padding: '2rem', textAlign: 'center' }}>
                <Spinner animation="border" role="status">
                  <span className="visually-hidden">Loading sources...</span>
                </Spinner>
              </div>
            ) : (
              <MultiModelFilesList
                files={multiModelFiles}
                sessionId={sessionId}
                userEmail={email}
                generatedDocuments={generatedDocuments}
                projectFiles={projectStructure?.files || []}
                onProjectFileClick={(file) => setViewingProjectFile(file)}
                viewingFile={viewingProjectFile}
              />
            )}
          </div>
        </div>
      )}

      {/* Show preview button when hidden */}
      {!showFilePreview && (
        <button
          onClick={() => setShowFilePreview(true)}
          style={{
            position: 'fixed',
            bottom: '2rem',
            right: '2rem',
            padding: '0.75rem 1rem',
            backgroundColor: '#2563eb',
            color: 'white',
            border: 'none',
            borderRadius: '50%',
            width: '56px',
            height: '56px',
            cursor: 'pointer',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            fontSize: '1.25rem',
            zIndex: 1000
          }}
          title="Show file preview"
        >
          📄
        </button>
      )}

      {showPremiumOverlay && <PremiumOverlay />}
      </div>
    </div>
  );
};

export default MultiAgent;