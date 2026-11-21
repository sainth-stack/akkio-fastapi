import React, { useEffect, useState } from "react";
import api from "../../utils/api";
import "./akkioUsageCard.scss";
import { FaChevronRight } from "react-icons/fa6";
import { BsLightningChargeFill } from "react-icons/bs";
import { IoCubeOutline } from "react-icons/io5";

export default function WorkspaceUsageCard() {
  const [usage, setUsage] = useState({
    credits_remaining: 100,
    storage_remaining_mb: 50,
  });

  useEffect(() => {
    let cancelled = false;

    async function loadUsage() {
      try {
        const resp = await api.get('/usage');
        const data = resp.data;
        if (!cancelled && data) {
          setUsage({
            credits_remaining: typeof data.credits_remaining === "number" ? data.credits_remaining : 100,
            storage_remaining_mb: typeof data.storage_remaining_mb === "number" ? data.storage_remaining_mb : 50,
          });
        }
      } catch (e) {
        console.error("Error loading workspace usage:", e);
      }
    }

    loadUsage();

    const interval = setInterval(loadUsage, 300000);
    const onFocus = () => loadUsage();
    const onUsageUpdate = () => loadUsage();

    window.addEventListener("focus", onFocus);
    window.addEventListener("usage_updated", onUsageUpdate);

    return () => {
      cancelled = true;
      clearInterval(interval);
      window.removeEventListener("focus", onFocus);
      window.removeEventListener("usage_updated", onUsageUpdate);
    };
  }, []);

  return (
    <div className="akkioUsageCard">
      <div className="akkioUsageCard__header">
        <span className="akkioUsageCard__title">Workspace Usage</span>
      </div>

      <div className="akkioUsageCard__rows">
        <div className="akkioUsageCard__row">
          <div className="akkioUsageCard__iconWrapper">
            <BsLightningChargeFill className="akkioUsageCard__icon" />
          </div>
          <div className="akkioUsageCard__content">
            <span className="akkioUsageCard__value">{usage.credits_remaining}</span>
            <span className="akkioUsageCard__label">AI Credits</span>
          </div>
        </div>

        <div className="akkioUsageCard__row">
          <div className="akkioUsageCard__iconWrapper">
            <IoCubeOutline className="akkioUsageCard__icon" />
          </div>
          <div className="akkioUsageCard__content">
            <span className="akkioUsageCard__value">{usage.storage_remaining_mb} <small>MB</small></span>
            <span className="akkioUsageCard__label">Storage Left</span>
          </div>
        </div>
      </div>

      <button className="akkioUsageCard__upgrade">
        <span>Upgrade Plan</span>
        <FaChevronRight size={12} />
      </button>
    </div>
  );
}
