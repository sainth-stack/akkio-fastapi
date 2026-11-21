import React from 'react';
import {
    IoSparkles,
    IoChevronDown,
    IoCheckmark,
    IoDocumentTextOutline,
    IoColorPaletteOutline,
    IoBrushOutline,
    IoLayersOutline,
    IoCodeSlashOutline,
    IoRocketOutline,
    IoHammerOutline,
    IoPeopleOutline,
    IoPhonePortraitOutline,
    IoFlaskOutline,
    IoSend,
    IoAppsOutline,
    IoCloudUploadOutline,
    IoGitBranchOutline,
    IoGlobeOutline,
    IoPlayOutline,
    IoStopCircleOutline,
    IoDownloadOutline,
    IoSparklesOutline,
    IoHardwareChipOutline,
    IoBulbOutline,
} from 'react-icons/io5';
import { FaPenToSquare, FaStar, FaPlus } from 'react-icons/fa6';
import { SiOpenai } from 'react-icons/si';
import { BsLightningChargeFill } from 'react-icons/bs';

export const IconBadge = ({ icon: Icon, variant = 'indigo', size = 14, className = '' }) => (
    <span className={`ab-icon-badge ab-icon-badge--${variant} ${className}`.trim()} aria-hidden>
        <Icon size={size} />
    </span>
);

export const TIER_ICONS = {
    flagship: IoSparklesOutline,
    balanced: IoHardwareChipOutline,
    fast: BsLightningChargeFill,
    reasoning: IoBulbOutline,
    legacy: IoCodeSlashOutline,
};

export {
    IoSparkles,
    IoChevronDown,
    IoCheckmark,
    IoDocumentTextOutline,
    IoColorPaletteOutline,
    IoBrushOutline,
    IoLayersOutline,
    IoCodeSlashOutline,
    IoRocketOutline,
    IoHammerOutline,
    IoPeopleOutline,
    IoPhonePortraitOutline,
    IoFlaskOutline,
    IoSend,
    IoAppsOutline,
    IoCloudUploadOutline,
    IoGitBranchOutline,
    IoGlobeOutline,
    IoPlayOutline,
    IoStopCircleOutline,
    IoDownloadOutline,
    FaPenToSquare,
    FaStar,
    FaPlus,
    SiOpenai,
};
