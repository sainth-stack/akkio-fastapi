import React from "react";
import { useNavigate } from "react-router-dom";

const containerStyle = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  minHeight: '70vh',
  textAlign: 'center',
  padding: '40px 20px'
};

const cardStyle = {
  backgroundColor: '#f8f9fa',
  borderRadius: '12px',
  padding: '60px 40px',
  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
  border: '2px dashed #dee2e6',
  maxWidth: '500px',
  width: '100%'
};

const iconStyle = {
  width: '80px',
  height: '80px',
  backgroundColor: '#e9ecef',
  borderRadius: '50%',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  margin: '0 auto 24px',
  fontSize: '32px',
  color: '#6c757d'
};

const headingStyle = {
  fontSize: '28px',
  fontWeight: '600',
  color: '#212529',
  marginBottom: '16px',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
};

const descStyle = {
  fontSize: '16px',
  color: '#6c757d',
  lineHeight: '1.6',
  marginBottom: '32px',
  maxWidth: '400px',
  margin: '0 auto 32px'
};

const buttonStyle = {
  backgroundColor: '#1976d2',
  color: 'white',
  border: 'none',
  borderRadius: '8px',
  padding: '12px 32px',
  fontSize: '16px',
  fontWeight: '500',
  cursor: 'pointer',
  transition: 'all 0.2s ease',
  boxShadow: '0 2px 8px rgba(25, 118, 210, 0.3)'
};

function EmptyState({
  title = 'No Data Uploaded',
  description = 'Upload your data file to start exploring insights, analyzing trends, and generating comprehensive reports.',
  ctaText = 'Upload Data',
  onCtaClick,
  navigateTo = '/data-source'
}) {
  const navigate = useNavigate();

  const handleClick = () => {
    if (typeof onCtaClick === 'function') {
      onCtaClick();
    } else if (navigateTo) {
      navigate(navigateTo);
    }
  };

  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        <div style={iconStyle}>📊</div>
        <h2 style={headingStyle}>{title}</h2>
        <p style={descStyle}>{description}</p>
        <button
          onClick={handleClick}
          style={buttonStyle}
          onMouseEnter={(e) => {
            e.target.style.backgroundColor = '#1565c0';
            e.target.style.transform = 'translateY(-1px)';
            e.target.style.boxShadow = '0 4px 12px rgba(25, 118, 210, 0.4)';
          }}
          onMouseLeave={(e) => {
            e.target.style.backgroundColor = '#1976d2';
            e.target.style.transform = 'translateY(0)';
            e.target.style.boxShadow = '0 2px 8px rgba(25, 118, 210, 0.3)';
          }}
        >
          {ctaText}
        </button>
      </div>
    </div>
  );
}

export default EmptyState;


