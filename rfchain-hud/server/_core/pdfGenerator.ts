/**
 * PDF Generator for Forensic Reports
 * Generates chain-of-custody documentation with hash verification
 */

import PDFDocument from "pdfkit";
import { PassThrough } from "stream";

interface ForensicReportData {
  // Analysis info
  analysisId: number;
  signalFilename: string;
  analysisDate: Date;
  
  // Signal metrics
  metrics: {
    avgPowerDbm: number;
    peakPowerDbm: number;
    paprDb: number;
    bandwidthHz: number;
    freqOffsetHz: number;
    iqImbalanceDb: number;
    snrEstimateDb: number;
    sampleCount: number;
    dcOffsetReal: number;
    dcOffsetImag: number;
  };
  
  // Analysis configuration
  config: {
    sampleRate: number;
    centerFreq: number;
    dataFormat: string;
    fftSize: number;
    analyzerVersion: string;
  };
  
  // Anomalies
  anomalies: {
    dcSpike: boolean;
    saturation: boolean;
    dropout: boolean;
    [key: string]: boolean;
  };
  
  // Hash chain
  hashChain: {
    stage: string;
    sha256: string | null;
    sha3_256: string | null;
    timestamp?: string;
  }[];
  
  // User info
  analyst: {
    name: string;
    id: string;
  };
}

/**
 * Generate a forensic report PDF
 */
export async function generateForensicPDF(data: ForensicReportData): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    try {
      const doc = new PDFDocument({
        size: "A4",
        margins: { top: 50, bottom: 50, left: 50, right: 50 },
        info: {
          Title: `Forensic Report - Analysis #${data.analysisId}`,
          Author: "RFChain Signal Intelligence System",
          Subject: "RF Signal Forensic Analysis Report",
          Keywords: "RF, forensic, signal analysis, chain of custody",
          CreationDate: new Date(),
        },
      });

      const chunks: Buffer[] = [];
      const stream = new PassThrough();
      
      doc.pipe(stream);
      
      stream.on("data", (chunk) => chunks.push(chunk));
      stream.on("end", () => resolve(Buffer.concat(chunks)));
      stream.on("error", reject);

      // Colors
      const primaryColor = "#00d4ff";
      const darkColor = "#0a1628";
      const textColor = "#333333";
      const lightGray = "#666666";

      // Header
      doc.rect(0, 0, doc.page.width, 100).fill(darkColor);
      
      doc.fontSize(24)
        .fillColor("#ffffff")
        .text("RFCHAIN", 50, 30, { continued: true })
        .fillColor(primaryColor)
        .text(" FORENSIC REPORT");
      
      doc.fontSize(10)
        .fillColor("#888888")
        .text("Signal Intelligence & Forensic Analysis Platform", 50, 60);
      
      doc.fontSize(10)
        .fillColor(primaryColor)
        .text(`Report ID: FR-${data.analysisId.toString().padStart(6, "0")}`, 400, 30)
        .text(`Generated: ${new Date().toISOString()}`, 400, 45)
        .text(`Classification: UNCLASSIFIED`, 400, 60);

      // Document title
      doc.moveDown(3);
      doc.fontSize(18)
        .fillColor(textColor)
        .text("Chain of Custody Documentation", { align: "center" });
      
      doc.moveDown(0.5);
      doc.fontSize(12)
        .fillColor(lightGray)
        .text(`Signal File: ${data.signalFilename}`, { align: "center" });

      // Section 1: Analysis Summary
      doc.moveDown(2);
      drawSectionHeader(doc, "1. ANALYSIS SUMMARY");
      
      doc.moveDown(0.5);
      doc.fontSize(10).fillColor(textColor);
      
      const summaryData = [
        ["Analysis ID", `#${data.analysisId}`],
        ["Signal Filename", data.signalFilename],
        ["Analysis Date", data.analysisDate.toISOString()],
        ["Analyst", `${data.analyst.name} (ID: ${data.analyst.id})`],
        ["Analyzer Version", data.config.analyzerVersion || "2.2.2-forensic"],
      ];
      
      drawTable(doc, summaryData, 50, doc.y);

      // Section 2: Signal Metrics
      doc.moveDown(2);
      drawSectionHeader(doc, "2. SIGNAL CHARACTERISTICS");
      
      doc.moveDown(0.5);
      
      const metricsData = [
        ["Average Power", `${data.metrics.avgPowerDbm.toFixed(2)} dBm`],
        ["Peak Power", `${data.metrics.peakPowerDbm.toFixed(2)} dBm`],
        ["PAPR", `${data.metrics.paprDb.toFixed(2)} dB`],
        ["Estimated Bandwidth", `${(data.metrics.bandwidthHz / 1000).toFixed(2)} kHz`],
        ["Frequency Offset", `${data.metrics.freqOffsetHz.toFixed(2)} Hz`],
        ["I/Q Imbalance", `${data.metrics.iqImbalanceDb.toFixed(3)} dB`],
        ["SNR Estimate", `${data.metrics.snrEstimateDb.toFixed(2)} dB`],
        ["Sample Count", data.metrics.sampleCount.toLocaleString()],
        ["DC Offset (Real)", data.metrics.dcOffsetReal.toFixed(6)],
        ["DC Offset (Imag)", data.metrics.dcOffsetImag.toFixed(6)],
      ];
      
      drawTable(doc, metricsData, 50, doc.y);

      // Section 3: Analysis Configuration
      doc.moveDown(2);
      drawSectionHeader(doc, "3. ANALYSIS CONFIGURATION");
      
      doc.moveDown(0.5);
      
      const configData = [
        ["Sample Rate", `${(data.config.sampleRate / 1e6).toFixed(2)} MHz`],
        ["Center Frequency", `${(data.config.centerFreq / 1e6).toFixed(2)} MHz`],
        ["Data Format", data.config.dataFormat],
        ["FFT Size", data.config.fftSize.toString()],
      ];
      
      drawTable(doc, configData, 50, doc.y);

      // Section 4: Anomaly Detection
      doc.moveDown(2);
      drawSectionHeader(doc, "4. ANOMALY DETECTION RESULTS");
      
      doc.moveDown(0.5);
      
      const anomalyEntries = Object.entries(data.anomalies);
      const hasAnomalies = anomalyEntries.some(([_, detected]) => detected);
      
      if (hasAnomalies) {
        doc.fontSize(10)
          .fillColor("#cc0000")
          .text("⚠ ANOMALIES DETECTED", 50, doc.y);
        doc.moveDown(0.5);
        
        anomalyEntries.forEach(([key, detected]) => {
          if (detected) {
            doc.fontSize(10)
              .fillColor(textColor)
              .text(`• ${formatAnomalyName(key)}: DETECTED`, 60, doc.y);
          }
        });
      } else {
        doc.fontSize(10)
          .fillColor("#00aa00")
          .text("✓ NO ANOMALIES DETECTED", 50, doc.y);
        doc.moveDown(0.5);
        doc.fontSize(10)
          .fillColor(textColor)
          .text("Signal passed all anomaly detection checks.", 50, doc.y);
      }

      // New page for hash chain
      doc.addPage();

      // Section 5: Forensic Hash Chain (Chain of Custody)
      drawSectionHeader(doc, "5. FORENSIC HASH CHAIN - CHAIN OF CUSTODY");
      
      doc.moveDown(0.5);
      doc.fontSize(9)
        .fillColor(lightGray)
        .text("The following cryptographic hashes verify the integrity of data at each processing stage.", 50, doc.y);
      doc.text("Any modification to the signal data would result in different hash values.", 50, doc.y + 12);
      
      doc.moveDown(2);

      // Hash chain table
      data.hashChain.forEach((checkpoint, index) => {
        const yPos = doc.y;
        
        // Stage header
        doc.fontSize(11)
          .fillColor(primaryColor)
          .text(`Stage ${index + 1}: ${formatStageName(checkpoint.stage)}`, 50, yPos);
        
        doc.moveDown(0.3);
        
        // SHA-256
        doc.fontSize(9)
          .fillColor(textColor)
          .text("SHA-256:", 60, doc.y);
        doc.fontSize(8)
          .fillColor(lightGray)
          .text(checkpoint.sha256 || "N/A", 120, doc.y - 10, { width: 400 });
        
        doc.moveDown(0.3);
        
        // SHA3-256
        doc.fontSize(9)
          .fillColor(textColor)
          .text("SHA3-256:", 60, doc.y);
        doc.fontSize(8)
          .fillColor(lightGray)
          .text(checkpoint.sha3_256 || "N/A", 120, doc.y - 10, { width: 400 });
        
        doc.moveDown(1);
        
        // Separator line
        if (index < data.hashChain.length - 1) {
          doc.moveTo(50, doc.y)
            .lineTo(545, doc.y)
            .strokeColor("#dddddd")
            .stroke();
          doc.moveDown(0.5);
        }
      });

      // Section 6: Verification Instructions
      doc.moveDown(2);
      drawSectionHeader(doc, "6. HASH VERIFICATION INSTRUCTIONS");
      
      doc.moveDown(0.5);
      doc.fontSize(9)
        .fillColor(textColor)
        .text("To verify the integrity of this analysis:", 50, doc.y);
      
      doc.moveDown(0.5);
      doc.text("1. Obtain the original signal file used for this analysis.", 60, doc.y);
      doc.text("2. Compute the SHA-256 hash of the raw signal data.", 60, doc.y + 12);
      doc.text("3. Compare with the 'raw_input' hash in the chain above.", 60, doc.y + 24);
      doc.text("4. A match confirms the signal data has not been modified.", 60, doc.y + 36);
      
      doc.moveDown(3);
      doc.fontSize(8)
        .fillColor(lightGray)
        .text("Linux/Mac: sha256sum <filename>", 60, doc.y);
      doc.text("Windows: certutil -hashfile <filename> SHA256", 60, doc.y + 10);

      // Footer on all pages
      const pages = doc.bufferedPageRange();
      for (let i = 0; i < pages.count; i++) {
        doc.switchToPage(i);
        
        // Footer line
        doc.moveTo(50, doc.page.height - 50)
          .lineTo(545, doc.page.height - 50)
          .strokeColor("#cccccc")
          .stroke();
        
        // Footer text
        doc.fontSize(8)
          .fillColor(lightGray)
          .text(
            `RFChain Forensic Report | Analysis #${data.analysisId} | Page ${i + 1} of ${pages.count}`,
            50,
            doc.page.height - 40,
            { align: "center", width: 495 }
          );
        
        doc.text(
          "This document is generated automatically and serves as chain-of-custody documentation.",
          50,
          doc.page.height - 30,
          { align: "center", width: 495 }
        );
      }

      doc.end();
    } catch (error) {
      reject(error);
    }
  });
}

function drawSectionHeader(doc: PDFKit.PDFDocument, title: string) {
  const primaryColor = "#00d4ff";
  
  doc.fontSize(12)
    .fillColor(primaryColor)
    .text(title, 50, doc.y);
  
  doc.moveTo(50, doc.y + 5)
    .lineTo(545, doc.y + 5)
    .strokeColor(primaryColor)
    .lineWidth(1)
    .stroke();
}

function drawTable(doc: PDFKit.PDFDocument, data: string[][], x: number, y: number) {
  const textColor = "#333333";
  const lightGray = "#666666";
  const colWidth = 245;
  
  let currentY = y;
  
  data.forEach(([label, value], index) => {
    // Alternate row background
    if (index % 2 === 0) {
      doc.rect(x, currentY - 2, 495, 16).fill("#f8f9fa");
    }
    
    doc.fontSize(10)
      .fillColor(lightGray)
      .text(label, x + 5, currentY, { width: colWidth - 10 });
    
    doc.fontSize(10)
      .fillColor(textColor)
      .text(value, x + colWidth, currentY, { width: colWidth - 10 });
    
    currentY += 18;
  });
  
  doc.y = currentY;
}

function formatStageName(stage: string): string {
  const stageNames: Record<string, string> = {
    raw_input: "Raw Input Data",
    post_metrics: "Post Metrics Calculation",
    post_anomaly: "Post Anomaly Detection",
    post_digital_analysis: "Post Digital Analysis",
    post_v3_analysis: "Post V3 Analysis",
    pre_output: "Pre-Output (Final)",
  };
  return stageNames[stage] || stage.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function formatAnomalyName(key: string): string {
  const names: Record<string, string> = {
    dc_spike: "DC Spike",
    saturation: "Signal Saturation",
    dropout: "Signal Dropout",
    clipping: "Signal Clipping",
    noise_floor: "Elevated Noise Floor",
  };
  return names[key] || key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

export type { ForensicReportData };
