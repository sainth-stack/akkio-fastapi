import React from 'react';
import { getFriendlyPipelineState } from './utils/errorMessages';
import PlanView from './PlanView';
import FileExplorer from './FileExplorer';
import AgentsView from './AgentsView';
import AppView from './AppView';
import DeploymentView from './DeploymentView';
import TestView from './TestView';
import Spinner from 'react-bootstrap/Spinner';
import {
    IoDocumentTextOutline,
    IoHammerOutline,
    IoRocketOutline,
    IoPeopleOutline,
    IoCodeSlashOutline,
    IoPhonePortraitOutline,
    IoFlaskOutline,
    IoStopCircleOutline,
    IoDownloadOutline,
    IoPlayOutline,
} from './AppBuilderIcons';

const MAIN_TABS = [
    { id: 'Plan', icon: IoDocumentTextOutline },
    { id: 'Build', icon: IoHammerOutline },
    { id: 'Deploy', icon: IoRocketOutline },
];

const BUILD_TABS = [
    { id: 'Multi Agents', icon: IoPeopleOutline },
    { id: 'Code', icon: IoCodeSlashOutline },
    { id: 'Build', icon: IoPhonePortraitOutline, label: 'App View' },
    { id: 'Test', icon: IoFlaskOutline },
];

const TabPanel = ({
    plan,
    prd,
    fileTree,
    files,
    activeTab,
    onTabChange,
    onRun,
    useSandboxBuild,
    onUseSandboxBuildChange,
    projectName,
    agents,
    logs,
    runState,
    appRefreshKey,
    onLoadFile,
    onSaveFile,
    currentPhase,
    onCreateAgents,
    onStartCodegen,
    onRestartCodegen,
    onRegeneratePrd,
    onGeneratePrd,
    requirementReady,
    onRegeneratePlan,
    onRegenerateAgents,
    isLoading,
    isCodegenLoading,
    isRunLoading,
    activeBuildTab,
    onBuildTabChange,
    // New Props for Manual Flow
    generatedUIUX,
    generatedArchitecture,
    onGenerateUIUX,
    onGenerateStyle,
    onGenerateArch,
    onGeneratePlan,
    designTokens,
    designSystemMd,
    streamingArchitectureText,
    onStopPlanning,
    onStopCodegen,
    onDownloadCode,
    onDownloadSampleData,
    // Deployment Props
    appId,
    isTreeLoading,
    buildStatus,
    buildError,
    buildLog,
    pipelineStatus,
    pipelineError,
    onRetryPipeline,
    builderKind,
    planJson,
    appTrack,
    selectedDomain,
}) => {
    const resolvedBuildTab = activeBuildTab || 'Multi Agents';
    const setResolvedBuildTab = onBuildTabChange || (() => { });

    const renderBuildContent = () => {
        const renderSubContent = () => {
            switch (resolvedBuildTab) {
                case 'Multi Agents':
                    return (
                        <AgentsView
                            agents={agents}
                            onRegenerate={onRegenerateAgents}
                            isLoading={isLoading}
                            selectedDomain={selectedDomain}
                        />
                    );
                case 'Code':
                    return (
                        <FileExplorer
                            tree={fileTree}
                            files={files}
                            projectName={projectName}
                            onLoadFile={onLoadFile}
                            onSaveFile={onSaveFile}
                            isTreeLoading={isTreeLoading}
                        />
                    );
                case 'Build': // App View
                    return (
                        <AppView
                            projectName={projectName}
                            runState={runState}
                            isRunLoading={isRunLoading}
                            logs={logs}
                            appRefreshKey={appRefreshKey}
                            buildStatus={buildStatus}
                            buildError={buildError}
                            buildLog={buildLog}
                            onRetryRun={onRun}
                        />
                    );
                case 'Test':
                    return (
                        <TestView
                            projectName={projectName}
                        />
                    );
                default:
                    return null;
            }
        };

        return (
            <div className="build-container">
                <div className="sub-tabs">
                    {BUILD_TABS.map(({ id, icon: TabIcon, label }) => (
                        <div
                            key={id}
                            className={`sub-tab ${resolvedBuildTab === id ? 'active' : ''}`}
                            onClick={() => setResolvedBuildTab(id)}
                        >
                            <TabIcon size={15} className="sub-tab-icon" aria-hidden />
                            {id === 'Multi Agents' && builderKind === 'fullstack' ? 'Coding Agents' : (label || id)}
                        </div>
                    ))}
                </div>
                <div className="sub-content">
                    {renderSubContent()}
                </div>
            </div>
        );
    };

    const renderContent = () => {
        switch (activeTab) {
            case 'Plan':
                return <PlanView
                    plan={plan}
                    prd={prd}
                    generatedUIUX={generatedUIUX}
                    generatedArchitecture={generatedArchitecture}
                    onGenerateUIUX={onGenerateUIUX}
                    onGenerateStyle={onGenerateStyle}
                    onGenerateArch={onGenerateArch}
                    onGeneratePlan={onGeneratePlan}
                    designTokens={designTokens}
                    designSystemMd={designSystemMd}
                    onRegeneratePrd={onRegeneratePrd}
                    onGeneratePrd={onGeneratePrd}
                    requirementReady={requirementReady}
                    onRegeneratePlan={onRegeneratePlan}
                    isLoading={isLoading}
                    streamingArchitectureText={streamingArchitectureText}
                    onStop={onStopPlanning}
                    pipelineStatus={pipelineStatus}
                    pipelineError={pipelineError}
                    onRetryPipeline={onRetryPipeline}
                    builderKind={builderKind}
                    planJson={planJson}
                    appTrack={appTrack}
                    selectedDomain={selectedDomain}
                />;
            case 'Build':
                return renderBuildContent();
            case 'Deploy':
                return (
                    <div className="deployment-tab-scroll">
                        <DeploymentView
                            projectName={projectName}
                            appId={appId}
                        />
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <div className="content-section">
            <div className="tabs-header">
                <div className="tabs-left">
                    {MAIN_TABS.map(({ id, icon: TabIcon }) => (
                        <div
                            key={id}
                            className={`tab ${activeTab === id ? 'active' : ''}`}
                            onClick={() => onTabChange(id)}
                        >
                            <TabIcon size={15} className="tab-icon" aria-hidden />
                            {id}
                        </div>
                    ))}
                </div>
                <div className="tabs-actions">
                    {/* Common actions can go here */}
                    {activeTab === 'Plan' && generatedArchitecture && (
                        <button
                            className="run-button"
                            onClick={onStartCodegen || onCreateAgents}
                            disabled={isLoading || isCodegenLoading}
                            style={{
                                backgroundColor: '#28a745',
                                padding: '8px 20px',
                                fontSize: '13px',
                                fontWeight: '600'
                            }}
                            title="Step 4 — run Coding, Integration, and Validator agents"
                        >
                            {(isLoading || isCodegenLoading) ? 'Generating code...' : 'Generate Code'}
                        </button>
                    )}

                    {activeTab === 'Plan' && isLoading && onStopPlanning && (
                        <button
                            type="button"
                            className="tab-action-btn tab-action-btn--danger"
                            onClick={onStopPlanning}
                        >
                            <IoStopCircleOutline size={15} />
                            Stop
                        </button>
                    )}

                    {activeTab === 'Build' && isCodegenLoading && onStopCodegen && (
                        <button
                            type="button"
                            className="tab-action-btn tab-action-btn--danger"
                            onClick={onStopCodegen}
                        >
                            <IoStopCircleOutline size={15} />
                            Stop Codegen
                        </button>
                    )}

                    {/* Show Run button in Build tab */}
                    {activeTab === 'Build' && (
                        <>
                            <label
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 6,
                                    fontSize: 12,
                                    color: '#475569',
                                    marginRight: 8,
                                    cursor: 'pointer',
                                    userSelect: 'none'
                                }}
                                title="Live sandbox preview (contact support to enable)"
                            >
                                <input
                                    type="checkbox"
                                    checked={!!useSandboxBuild}
                                    onChange={(e) => onUseSandboxBuildChange && onUseSandboxBuildChange(e.target.checked)}
                                />
                                E2B sandbox
                            </label>
                            <button
                                className="download-button tab-action-btn tab-action-btn--secondary"
                                onClick={onDownloadCode}
                                disabled={!projectName}
                            >
                                <IoDownloadOutline size={15} />
                                Download Code
                            </button>
                            <button
                                className="download-button tab-action-btn tab-action-btn--secondary"
                                onClick={onDownloadSampleData}
                                disabled={!projectName || !onDownloadSampleData}
                                title="Download synthetic demo data (KPIs, Detail rows, Breakdown) as Excel"
                            >
                                <IoDownloadOutline size={15} />
                                Download synthetic data
                            </button>
                            <button
                                className="run-button tab-action-btn tab-action-btn--primary"
                                onClick={onRun}
                                disabled={!projectName || isRunLoading}
                            >
                                {isRunLoading ? <Spinner animation="border" size="sm" /> : <IoPlayOutline size={15} />}
                                {isRunLoading ? 'Starting...' : 'Run App'}
                            </button>
                            {buildStatus === 'BUILD_FAILED' && onRun && (
                                <button
                                    type="button"
                                    className="run-button"
                                    onClick={onRun}
                                    disabled={isRunLoading}
                                    style={{
                                        marginLeft: '10px',
                                        backgroundColor: '#dc3545',
                                        padding: '8px 16px',
                                        fontSize: '13px',
                                        fontWeight: '600',
                                    }}
                                >
                                    Retry Run
                                </button>
                            )}
                        </>
                    )}
                </div>
            </div>
            <div className="tab-content">
                {renderContent()}
            </div>
        </div>
    );
};

export default TabPanel;
