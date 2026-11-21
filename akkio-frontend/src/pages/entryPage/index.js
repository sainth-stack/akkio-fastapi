// MiddleContent.js

import React from 'react';
import { Link } from "react-router-dom";
import styles from './middle.module.scss';
const MiddleContent = () => {
    return (
        <div className=' p-0 m-0'>
            <div className='row'>
                <div className={`${styles.middleContent} col-6`} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: '92vh', justifyContent: 'center' }}>
                    <h1>Welcome to Otamat Platform</h1>
                    <p style={{ width: '90%' }}>
                        Decision-based Autonomous GenAI AgenticAI Platform" combines elements of decision support, Autonomous Capability, advanced analytics, data-driven insights, and Generative Artificial intelligence to empower users to make informed decisions and optimize processes across various domains.
                    </p>
                    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', justifyContent: 'center' }}>
                        <Link to={'/app-builder'} className="nav-link align-middle px-2 nav-item">
                            <button className={styles.ctaButton}>App Builder</button>
                        </Link>
                        <Link to={'/agentic-builder'} className="nav-link align-middle px-2 nav-item">
                            <button className={styles.ctaButton}>Agentic Builder</button>
                        </Link>
                    </div>
                </div>
                <div className={`${styles.rightContent} col-6`}>

                </div>
            </div>
        </div>
    );
};

export default MiddleContent;