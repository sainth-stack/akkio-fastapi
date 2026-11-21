import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FaCrown } from 'react-icons/fa';
import './PremiumOverlay.scss';

const PremiumOverlay = ({ message = "You've run out of AI credits." }) => {
  const navigate = useNavigate();

  return (
    <div className="premium-overlay-container">
      <div className="premium-overlay-content">
        <div className="icon-container">
          <FaCrown className="crown-icon" />
        </div>
        <h2>Upgrade to Premium</h2>
        <p>{message}</p>
        <p className="sub-text">Upgrade your plan to continue using advanced AI features.</p>
        <button 
          className="upgrade-btn"
          onClick={() => navigate('/welcome')}
        >
          View Plans & Upgrade
        </button>
      </div>
    </div>
  );
};

export default PremiumOverlay;


