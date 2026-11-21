import React, { useState, useCallback } from 'react';
import { Input, Button, Upload, message, Progress, Alert, Space, Tag, Slider, Tooltip, Collapse } from 'antd';
import { InboxOutlined, DeleteOutlined, FileTextOutlined, FilePdfOutlined, FileImageOutlined, FileExcelOutlined, CheckCircleOutlined } from '@ant-design/icons';
import { FaMagic } from 'react-icons/fa';
import api from '../../utils/api';


const { Dragger } = Upload;
const { TextArea } = Input;

export const MultiModelTraining = ({ onTrainingComplete, initialData }) => {
  const [modelName, setModelName] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [fileList, setFileList] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [temperature, setTemperature] = useState(0.0);
  const [workflow, setWorkflow] = useState('');
  const [outputFormat, setOutputFormat] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);


  React.useEffect(() => {
    if (initialData) {
      setModelName(initialData.model_name || '');
      setSystemPrompt(initialData.system_prompt || '');
      setTemperature(initialData.temperature || 0.0);
      setWorkflow(initialData.workflow || '');
      setOutputFormat(initialData.output_format || '');
    }
  }, [initialData]);

  const getFileIcon = (fileName) => {
    const ext = fileName.toLowerCase().split('.').pop();
    if (['pdf'].includes(ext)) return <FilePdfOutlined style={{ color: '#f40' }} />;
    if (['jpg', 'jpeg', 'png', 'gif', 'webp'].includes(ext)) return <FileImageOutlined style={{ color: '#52c41a' }} />;
    if (['csv', 'xlsx', 'xls'].includes(ext)) return <FileExcelOutlined style={{ color: '#1890ff' }} />;
    if (['mp3', 'wav', 'ogg', 'm4a', 'mp4'].includes(ext)) return <span style={{ fontSize: 16 }}>🎵</span>;
    return <FileTextOutlined />;
  };

  const getFileType = (fileName) => {
    const ext = fileName.toLowerCase().split('.').pop();
    if (['csv', 'xlsx', 'xls'].includes(ext)) return 'tabular';
    if (['pdf', 'doc', 'docx', 'txt'].includes(ext)) return 'document';
    if (['jpg', 'jpeg', 'png', 'gif', 'webp'].includes(ext)) return 'image';
    if (['mp3', 'wav', 'ogg', 'm4a', 'mp4'].includes(ext)) return 'audio';
    return 'other';
  };

  const handleFileChange = (info) => {
    // Filter out duplicates and system files
    const filteredFileList = info.fileList.filter((file, index, self) => {
      const fileName = file.name.toLowerCase();
      const isDuplicate = self.findIndex(f => f.name === file.name) !== index;
      const isSystemFile = fileName.startsWith('.ds_store') ||
        fileName.startsWith('thumbs.db') ||
        fileName.startsWith('desktop.ini') ||
        fileName.startsWith('.');
      return !isDuplicate && !isSystemFile;
    });

    const newFileList = filteredFileList.map(file => ({
      ...file,
      status: 'done',
      type: getFileType(file.name)
    }));

    setFileList(newFileList);
  };

  const handleRemoveFile = (file) => {
    setFileList(prev => prev.filter(f => f.uid !== file.uid));
  };


  const handleGenerateBackground = async () => {
    if (!systemPrompt.trim()) {
      message.error("Please enter a System Prompt first to generate background info.");
      return;
    }

    setIsGenerating(true);
    message.loading({ content: 'Generating background & output format...', key: 'generating' });

    try {
      const response = await api.post('/multi-model/generate_background', {
        model_name: modelName || "New Agent", // Fallback if empty, though backend might use it
        system_prompt: systemPrompt
      });

      if (response.data.status === 'success') {
        const { background, output_format } = response.data;
        setWorkflow(background);
        setOutputFormat(output_format);
        message.success({ content: 'Generated successfully!', key: 'generating' });
      } else {
        message.error({ content: 'Failed to generate content.', key: 'generating' });
      }
    } catch (error) {
      console.error("Generation error:", error);
      message.error({ content: 'Error creating AI content. Please try again.', key: 'generating' });
    } finally {
      setIsGenerating(false);
    }
  };




  const handleTrain = async () => {
    // Validation
    if (!modelName.trim()) {
      message.error('Please enter a model name');
      return;
    }

    setIsTraining(true);
    message.loading({ content: initialData?.session_id ? 'Updating model...' : 'Creating model...', key: 'training' });

    try {
      const isEdit = Boolean(initialData?.session_id);
      const hasFiles = fileList.length > 0;

      // EDIT path (no files): update config only
      if (isEdit && !hasFiles) {
        const resp = await api.post('/multi-model/update-config', {
          session_id: initialData.session_id,
          model_name: modelName.trim(),
          system_prompt: systemPrompt.trim(),
          temperature,
          workflow,
          output_format: outputFormat
        });

        message.success({ content: 'Model updated successfully!', key: 'training' });
        handleReset();

        if (onTrainingComplete) {
          onTrainingComplete({
            status: 'success',
            ...resp.data
          });
        }
        return;
      }

      // CREATE (or retrain overwrite) path: synchronous create
      const formData = new FormData();
      formData.append('model_name', modelName.trim());
      formData.append('system_prompt', systemPrompt.trim());
      formData.append('temperature', temperature);
      formData.append('workflow', workflow);
      formData.append('output_format', outputFormat);

      const fileTypes = {};
      fileList.forEach((file) => {
        formData.append('files', file.originFileObj || file);
        fileTypes[file.name] = getFileType(file.name);
      });
      formData.append('file_types', JSON.stringify(fileTypes));

      const response = await api.post('/multi-model/create', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (isEdit && initialData.session_id && initialData.session_id !== response.data?.session_id) {
        try {
          await api.delete('/multi-model/delete', {
            params: { session_id: initialData.session_id }
          });
        } catch (deleteError) {
          console.warn('Could not remove previous agent version:', deleteError);
        }
      }

      message.success({ content: isEdit ? 'Model updated successfully!' : 'Model created successfully!', key: 'training' });
      handleReset();

      if (onTrainingComplete) {
        onTrainingComplete({
          status: 'success',
          ...response.data
        });
      }
    } catch (error) {
      console.error('Create/Update error:', error);
      let errorMsg = initialData?.session_id ? 'Failed to update model' : 'Failed to create model';
      if (error.response?.data?.detail) {
        const detail = error.response.data.detail;
        errorMsg = typeof detail === 'object' ? JSON.stringify(detail) : detail;
      }
      message.error({ content: errorMsg, key: 'training' });
      setIsTraining(false);
    }
  };

  const handleReset = () => {
    setModelName('');
    setSystemPrompt('');
    setFileList([]);
    setTemperature(0.0);
    setWorkflow('');
    setOutputFormat('');
    setUploadProgress(0);
    setIsTraining(false);
  };

  const getFileSummary = () => {
    const summary = {
      tabular: 0,
      document: 0,
      image: 0,
      audio: 0,
      other: 0
    };
    fileList.forEach(file => {
      summary[file.type] = (summary[file.type] || 0) + 1;
    });
    return summary;
  };

  const fileSummary = getFileSummary();

  return (
    <div style={{
      background: '#fff',
      padding: 0,
      borderRadius: 12,
      border: '1px solid #e5e7eb',
      marginBottom: 0,
      fontFamily: "'Inter', sans-serif",
      height: '80vh',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{ padding: '24px 24px 0 24px' }}>
        <h2 style={{ fontSize: 20, fontWeight: 600, color: '#111827', marginBottom: 24 }}>
          Create Multi-Domain AI Model
        </h2>
      </div>

      {/* Scrollable Content */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '0 24px',
        display: 'flex',
        flexDirection: 'column',
        gap: 24
      }}>
        {/* Model Name */}
        <div>
          <label style={{ display: 'block', marginBottom: 8, fontWeight: 500, color: '#374151', fontSize: 14 }}>
            Application Name <span style={{ color: '#ef4444' }}>*</span>
          </label>
          <Input
            placeholder="e.g. Finance_Q3_Analysis_Agent"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            size="large"
            disabled={isTraining}
            style={{ borderRadius: 6, fontSize: 14 }}
          />
        </div>

        {/* System Prompt */}
        <div>
          <label style={{ display: 'block', marginBottom: 8, fontWeight: 500, color: '#374151', fontSize: 14 }}>
            Input
          </label>
          <TextArea
            placeholder="Define the persona and capabilities of your AI agent..."
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={5}
            size="large"
            disabled={isTraining}
            maxLength={2000}
            showCount
            style={{ borderRadius: 6, fontSize: 14 }}
          />
          <div style={{ marginTop: 6, fontSize: 12, color: '#6b7280' }}>
            Define the AI's role, capabilities, and behavior. (Displayed as Description in list)
          </div>
        </div>

        {/* Temperature */}
        <div>
          <label style={{ display: 'block', marginBottom: 8, fontWeight: 500, color: '#374151', fontSize: 14 }}>
            Creativity (Temperature): {temperature}
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <Slider
              min={0}
              max={1.0}
              step={0.1}
              value={temperature}
              onChange={setTemperature}
              disabled={isTraining}
              style={{ flex: 1 }}
              trackStyle={{ backgroundColor: '#4f46e5' }}
              handleStyle={{ borderColor: '#4f46e5', backgroundColor: '#4f46e5' }}
            />
          </div>
          <div style={{ marginTop: 4, fontSize: 12, color: '#6b7280' }}>
            0.0 = Deterministic/Focused, 1.0 = Creative/Random
          </div>
        </div>
        {/* Workflow Accordion */}
        <div style={{ marginBottom: 24 }}>
          <Collapse
            ghost
            expandIconPosition="end"
            items={[{
              key: '1',
              label: <span style={{ fontWeight: 500, color: '#374151', fontSize: 14 }}>Background</span>,
              children: (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <Button
                      icon={<FaMagic />}
                      onClick={handleGenerateBackground}
                      loading={isGenerating}
                      disabled={isTraining || !systemPrompt.trim()}
                      size="small"
                      style={{
                        color: '#4f46e5',
                        borderColor: '#e0e7ff',
                        background: '#eef2ff',
                        display: 'flex',
                        alignItems: 'center',
                        fontWeight: 500
                      }}
                    >
                      Auto-Generate
                    </Button>
                  </div>
                  {/* Background Information */}
                  <div>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 500, color: '#374151', fontSize: 14 }}>
                      Background Information
                    </label>
                    <TextArea
                      placeholder="Provide background context, instructions, and guidelines..."
                      value={workflow}
                      onChange={(e) => setWorkflow(e.target.value)}
                      rows={8}
                      disabled={isTraining}
                      style={{ borderRadius: 6, fontSize: 14 }}
                    />
                  </div>

                  {/* Output Format */}
                  <div>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 500, color: '#374151', fontSize: 14 }}>
                      Output Format
                    </label>
                    <TextArea
                      placeholder="Describe the expected output format..."
                      value={outputFormat}
                      onChange={(e) => setOutputFormat(e.target.value)}
                      rows={3}
                      disabled={isTraining}
                      style={{ borderRadius: 6, fontSize: 14 }}
                    />
                  </div>
                </div>
              )
            }]}
          />
        </div>

        {/* File Upload */}
        <div>
          <label style={{ display: 'block', marginBottom: 8, fontWeight: 500, color: '#374151', fontSize: 14 }}>
            Upload Data
          </label>
          <Dragger
            multiple
            fileList={fileList}
            onChange={handleFileChange}
            beforeUpload={() => false}
            accept=".csv,.xlsx,.xls,.pdf,.doc,.docx,.txt,image/*,.mp3,.wav,.ogg,.m4a,.mp4"
            disabled={isTraining}
            showUploadList={false}
            style={{
              borderRadius: 12,
              background: '#f9fafb',
              border: '1px dashed #d1d5db',
              padding: '32px 0'
            }}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined style={{ fontSize: 40, color: '#4f46e5' }} />
            </p>
            <p className="ant-upload-text" style={{ fontSize: 16, fontWeight: 500, color: '#111827' }}>
              Click or drag files to upload
            </p>
            <p className="ant-upload-hint" style={{ color: '#6b7280', fontSize: 13 }}>
              Support for CSV, Excel, PDF, Word, Images, Audio (max 100 files)
            </p>
          </Dragger>
        </div>

        {/* File Summary */}
        {
          fileList.length > 0 && (
            <div className="animate-fade-in" style={{ animation: 'fadeIn 0.3s ease-in-out' }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 12
              }}>
                <label style={{ fontWeight: 600, color: '#374151', fontSize: 14 }}>
                  Files ({fileList.length})
                </label>
                <Space size="small">
                  {fileSummary.tabular > 0 && <Tag color="blue" style={{ borderRadius: 12 }}>📊 {fileSummary.tabular}</Tag>}
                  {fileSummary.document > 0 && <Tag color="red" style={{ borderRadius: 12 }}>📄 {fileSummary.document}</Tag>}
                  {fileSummary.image > 0 && <Tag color="green" style={{ borderRadius: 12 }}>🖼️ {fileSummary.image}</Tag>}
                  {fileSummary.audio > 0 && <Tag color="purple" style={{ borderRadius: 12 }}>🎵 {fileSummary.audio}</Tag>}
                </Space>
              </div>

              <div style={{
                maxHeight: 220,
                overflowY: 'auto',
                border: '1px solid #e5e7eb',
                borderRadius: 8,
                background: '#fff'
              }}>
                {fileList.map((file, index) => (
                  <div
                    key={file.uid}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                      padding: '10px 16px',
                      borderBottom: index === fileList.length - 1 ? 'none' : '1px solid #f3f4f6',
                      transition: 'background 0.2s'
                    }}
                    className="file-list-item"
                  >
                    <div style={{ fontSize: 18, color: '#6b7280', display: 'flex' }}>
                      {getFileIcon(file.name)}
                    </div>
                    <div style={{
                      flex: 1,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      fontSize: 14,
                      color: '#374151'
                    }}>
                      {file.name}
                    </div>
                    <div style={{ fontSize: 12, color: '#9ca3af', minWidth: 60, textAlign: 'right' }}>
                      {(file.size / 1024).toFixed(0)} KB
                    </div>
                    <Button
                      type="text"
                      danger
                      icon={<DeleteOutlined style={{ fontSize: 14 }} />}
                      onClick={() => handleRemoveFile(file)}
                      disabled={isTraining}
                      size="small"
                      style={{ borderRadius: 4 }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )
        }
      </div>

      {/* Actions Footer */}
      <div style={{
        padding: '16px 24px',
        borderTop: '1px solid #e5e7eb',
        display: 'flex',
        justifyContent: 'flex-end',
        gap: 12,
        background: '#fff',
        borderRadius: '0 0 12px 12px'
      }}>
        <Button onClick={handleReset} disabled={isTraining} style={{ borderRadius: 6, fontWeight: 500 }}>
          Reset
        </Button>
        <Button
          type="primary"
          loading={isTraining}
          onClick={handleTrain}
          disabled={!modelName.trim()}
          style={{
            background: '#4f46e5',
            borderColor: '#4f46e5',
            borderRadius: 6,
            color: 'white',
            fontWeight: 600,
            padding: '0 24px',
            height: 36,
            boxShadow: '0 2px 4px rgba(79, 70, 229, 0.2)'
          }}
        >
          {isTraining ? 'Creating...' : 'Create Model'}
        </Button>
      </div>
    </div >
  );
};

export default MultiModelTraining;

