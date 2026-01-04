import { JarvisLayout } from "@/components/JarvisLayout";
import { JarvisPanel, JarvisStat, JarvisStatusIndicator } from "@/components/JarvisPanel";
import { Radio, Activity, Waves, BarChart3, Eye, ChevronRight, AlertTriangle, CheckCircle } from "lucide-react";
import { trpc } from "@/lib/trpc";
import { Link, useParams } from "wouter";
import { Button } from "@/components/ui/button";
import { useMemo } from "react";

export default function Analysis() {
  const params = useParams<{ id?: string }>();
  const analysisId = params.id ? parseInt(params.id, 10) : undefined;

  // If we have an ID, show the detail view
  if (analysisId) {
    return <AnalysisDetail id={analysisId} />;
  }

  // Otherwise show the list view
  return <AnalysisList />;
}

function AnalysisList() {
  const { data: analyses, isLoading } = trpc.analysis.getRecent.useQuery({ limit: 20 });

  return (
    <JarvisLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-wider jarvis-text">SIGNAL ANALYSIS</h2>
            <p className="text-sm text-muted-foreground mt-1">
              View and explore RF signal analysis results
            </p>
          </div>
          <Link href="/upload" className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring bg-primary text-primary-foreground shadow hover:bg-primary/90 h-9 px-4 py-2"><Radio className="w-4 h-4" />New Analysis</Link>
        </div>

        {/* Analysis List */}
        <JarvisPanel title="Analysis History" scanLine>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="h-20 bg-secondary/50 rounded animate-pulse" />
              ))}
            </div>
          ) : analyses && analyses.length > 0 ? (
            <div className="space-y-3">
              {analyses.map((analysis) => (
                <Link 
                  key={analysis.id} 
                  href={`/analysis/${analysis.id}`}
                  className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg border border-border/30 hover:border-primary/50 transition-all group"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded bg-primary/20 flex items-center justify-center">
                      <Radio className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium">Analysis #{analysis.id}</p>
                      <div className="flex items-center gap-4 mt-1">
                        <span className="text-xs text-muted-foreground">
                          {new Date(analysis.createdAt).toLocaleString()}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {analysis.sampleCount?.toLocaleString() || 0} samples
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <JarvisStatusIndicator
                      status="online"
                      label="COMPLETED"
                    />
                    <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
                  </div>
                </Link>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <Radio className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium">No analyses yet</p>
              <p className="text-sm text-muted-foreground mt-1">
                Upload a signal file to begin forensic analysis
              </p>
              <Link href="/upload" className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring bg-primary text-primary-foreground shadow hover:bg-primary/90 h-9 px-4 py-2 mt-4">Upload Signal</Link>
            </div>
          )}
        </JarvisPanel>
      </div>
    </JarvisLayout>
  );
}

function AnalysisDetail({ id }: { id: number }) {
  const { data: analysis, isLoading } = trpc.analysis.getById.useQuery({ id });

  // Parse the fullMetrics JSON if available
  const fullMetrics = useMemo(() => {
    if (analysis?.fullMetrics) {
      try {
        return typeof analysis.fullMetrics === 'string' 
          ? JSON.parse(analysis.fullMetrics) 
          : analysis.fullMetrics;
      } catch {
        return null;
      }
    }
    return null;
  }, [analysis?.fullMetrics]);

  // Parse anomalies
  const anomalies = useMemo(() => {
    if (analysis?.anomalies) {
      try {
        return typeof analysis.anomalies === 'string' 
          ? JSON.parse(analysis.anomalies) 
          : analysis.anomalies;
      } catch {
        return null;
      }
    }
    return null;
  }, [analysis?.anomalies]);

  // Get plot URLs from fullMetrics if available
  const plotUrls = useMemo(() => {
    if (fullMetrics?.plot_urls) {
      // The plot_urls object has keys like "filename_01_time_domain.png" -> "/analysis_output/id/filename_01_time_domain.png"
      // We need to find the right URL by matching the pattern
      const urls = fullMetrics.plot_urls;
      const findPlot = (pattern: string) => {
        const key = Object.keys(urls).find(k => k.includes(pattern));
        return key ? urls[key] : null;
      };
      return {
        time_domain: findPlot('01_time_domain'),
        frequency_domain: findPlot('02_frequency_domain'),
        spectrogram: findPlot('03_spectrogram'),
        waterfall: findPlot('04_waterfall'),
        constellation: findPlot('05_constellation'),
        phase: findPlot('06_phase'),
        autocorrelation: findPlot('07_autocorrelation'),
        cyclostationary: findPlot('08_cyclostationary'),
        statistics: findPlot('09_statistics'),
        power_analysis: findPlot('10_power_analysis'),
        eye_diagram: findPlot('11_eye_diagram'),
        digital_analysis: findPlot('12_digital_analysis'),
      };
    }
    return {};
  }, [fullMetrics]);

  if (isLoading) {
    return (
      <JarvisLayout>
        <div className="space-y-6">
          <div className="h-8 w-64 bg-secondary/50 rounded animate-pulse" />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-64 bg-secondary/50 rounded animate-pulse" />
            ))}
          </div>
        </div>
      </JarvisLayout>
    );
  }

  if (!analysis) {
    return (
      <JarvisLayout>
        <div className="text-center py-12">
          <Radio className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-50" />
          <p className="text-lg font-medium">Analysis not found</p>
          <p className="text-sm text-muted-foreground mt-1">
            The requested analysis could not be found
          </p>
          <Link href="/analysis" className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring bg-primary text-primary-foreground shadow hover:bg-primary/90 h-9 px-4 py-2 mt-4">Back to List</Link>
        </div>
      </JarvisLayout>
    );
  }

  const hasAnomalies = anomalies && (anomalies.dc_spike || anomalies.saturation || anomalies.dropout);

  return (
    <JarvisLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
              <Link href="/analysis" className="hover:text-primary">Analysis</Link>
              <ChevronRight className="w-4 h-4" />
              <span>Detail</span>
            </div>
            <h2 className="text-2xl font-bold tracking-wider jarvis-text">
              SIGNAL ANALYSIS #{id}
            </h2>
          </div>
          <Link href={`/forensics/${id}`} className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring border border-input bg-transparent shadow-sm hover:bg-accent hover:text-accent-foreground h-9 px-4 py-2"><Eye className="w-4 h-4" />View Forensic Report</Link>
        </div>

        {/* Anomaly Alert */}
        {hasAnomalies && (
          <JarvisPanel glowPulse>
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-6 h-6 text-yellow-500 flex-shrink-0" />
              <div>
                <h4 className="font-semibold text-yellow-500">Anomalies Detected</h4>
                <ul className="text-sm text-muted-foreground mt-1 space-y-1">
                  {anomalies.dc_spike && <li>• DC spike detected in signal</li>}
                  {anomalies.saturation && <li>• Signal saturation (clipping) detected</li>}
                  {anomalies.dropout && <li>• Signal dropout detected</li>}
                  {anomalies.details?.map((detail: string, i: number) => (
                    <li key={i}>• {detail}</li>
                  ))}
                </ul>
              </div>
            </div>
          </JarvisPanel>
        )}

        {/* Metrics Overview */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <JarvisPanel>
            <JarvisStat
              label="Avg Power"
              value={analysis.avgPowerDbm?.toFixed(2) ?? "N/A"}
              unit="dBm"
            />
          </JarvisPanel>
          <JarvisPanel>
            <JarvisStat
              label="Peak Power"
              value={analysis.peakPowerDbm?.toFixed(2) ?? "N/A"}
              unit="dBm"
            />
          </JarvisPanel>
          <JarvisPanel>
            <JarvisStat
              label="PAPR"
              value={analysis.paprDb?.toFixed(2) ?? "N/A"}
              unit="dB"
            />
          </JarvisPanel>
          <JarvisPanel>
            <JarvisStat
              label="SNR Estimate"
              value={analysis.snrEstimateDb?.toFixed(2) ?? "N/A"}
              unit="dB"
              status={analysis.snrEstimateDb && analysis.snrEstimateDb > 10 ? "good" : "warning"}
            />
          </JarvisPanel>
        </div>

        {/* Visualizations Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Time Domain */}
          <JarvisPanel title="Time Domain" scanLine>
            <div className="aspect-video bg-secondary/30 rounded-lg flex items-center justify-center overflow-hidden">
              {plotUrls.time_domain ? (
                <img
                  src={plotUrls.time_domain}
                  alt="Time Domain"
                  className="w-full h-full object-contain rounded-lg"
                />
              ) : (
                <div className="text-center">
                  <Activity className="w-12 h-12 text-muted-foreground mx-auto mb-2 opacity-50" />
                  <p className="text-sm text-muted-foreground">No visualization available</p>
                </div>
              )}
            </div>
          </JarvisPanel>

          {/* Frequency Domain */}
          <JarvisPanel title="Frequency Domain" scanLine>
            <div className="aspect-video bg-secondary/30 rounded-lg flex items-center justify-center overflow-hidden">
              {plotUrls.frequency_domain ? (
                <img
                  src={plotUrls.frequency_domain}
                  alt="Frequency Domain"
                  className="w-full h-full object-contain rounded-lg"
                />
              ) : (
                <div className="text-center">
                  <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-2 opacity-50" />
                  <p className="text-sm text-muted-foreground">No visualization available</p>
                </div>
              )}
            </div>
          </JarvisPanel>

          {/* Spectrogram */}
          <JarvisPanel title="Spectrogram" scanLine>
            <div className="aspect-video bg-secondary/30 rounded-lg flex items-center justify-center overflow-hidden">
              {plotUrls.spectrogram ? (
                <img
                  src={plotUrls.spectrogram}
                  alt="Spectrogram"
                  className="w-full h-full object-contain rounded-lg"
                />
              ) : (
                <div className="text-center">
                  <Waves className="w-12 h-12 text-muted-foreground mx-auto mb-2 opacity-50" />
                  <p className="text-sm text-muted-foreground">No visualization available</p>
                </div>
              )}
            </div>
          </JarvisPanel>

          {/* Constellation */}
          <JarvisPanel title="Constellation Diagram" scanLine>
            <div className="aspect-video bg-secondary/30 rounded-lg flex items-center justify-center overflow-hidden">
              {plotUrls.constellation ? (
                <img
                  src={plotUrls.constellation}
                  alt="Constellation"
                  className="w-full h-full object-contain rounded-lg"
                />
              ) : (
                <div className="text-center">
                  <Radio className="w-12 h-12 text-muted-foreground mx-auto mb-2 opacity-50" />
                  <p className="text-sm text-muted-foreground">No visualization available</p>
                </div>
              )}
            </div>
          </JarvisPanel>
        </div>

        {/* Additional Plots if available */}
        {(plotUrls.waterfall || plotUrls.phase || plotUrls.autocorrelation || plotUrls.cyclostationary || plotUrls.statistics || plotUrls.power_analysis || plotUrls.eye_diagram) && (
          <JarvisPanel title="Additional Visualizations">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {plotUrls.waterfall && (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Waterfall</p>
                  <div className="aspect-video bg-secondary/30 rounded-lg overflow-hidden">
                    <img src={plotUrls.waterfall} alt="Waterfall" className="w-full h-full object-contain" />
                  </div>
                </div>
              )}
              {plotUrls.phase && (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Phase</p>
                  <div className="aspect-video bg-secondary/30 rounded-lg overflow-hidden">
                    <img src={plotUrls.phase} alt="Phase" className="w-full h-full object-contain" />
                  </div>
                </div>
              )}
              {plotUrls.autocorrelation && (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Autocorrelation</p>
                  <div className="aspect-video bg-secondary/30 rounded-lg overflow-hidden">
                    <img src={plotUrls.autocorrelation} alt="Autocorrelation" className="w-full h-full object-contain" />
                  </div>
                </div>
              )}
              {plotUrls.cyclostationary && (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Cyclostationary</p>
                  <div className="aspect-video bg-secondary/30 rounded-lg overflow-hidden">
                    <img src={plotUrls.cyclostationary} alt="Cyclostationary" className="w-full h-full object-contain" />
                  </div>
                </div>
              )}
              {plotUrls.statistics && (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Statistics</p>
                  <div className="aspect-video bg-secondary/30 rounded-lg overflow-hidden">
                    <img src={plotUrls.statistics} alt="Statistics" className="w-full h-full object-contain" />
                  </div>
                </div>
              )}
              {plotUrls.power_analysis && (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Power Analysis</p>
                  <div className="aspect-video bg-secondary/30 rounded-lg overflow-hidden">
                    <img src={plotUrls.power_analysis} alt="Power Analysis" className="w-full h-full object-contain" />
                  </div>
                </div>
              )}
              {plotUrls.eye_diagram && (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Eye Diagram</p>
                  <div className="aspect-video bg-secondary/30 rounded-lg overflow-hidden">
                    <img src={plotUrls.eye_diagram} alt="Eye Diagram" className="w-full h-full object-contain" />
                  </div>
                </div>
              )}
              {plotUrls.digital_analysis && (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Digital Analysis</p>
                  <div className="aspect-video bg-secondary/30 rounded-lg overflow-hidden">
                    <img src={plotUrls.digital_analysis} alt="Digital Analysis" className="w-full h-full object-contain" />
                  </div>
                </div>
              )}
            </div>
          </JarvisPanel>
        )}

        {/* Signal Characteristics */}
        <JarvisPanel title="Signal Characteristics">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <JarvisStat
              label="Sample Count"
              value={analysis.sampleCount?.toLocaleString() ?? "N/A"}
            />
            <JarvisStat
              label="Duration"
              value={analysis.durationMs?.toFixed(2) ?? "N/A"}
              unit="ms"
            />
            <JarvisStat
              label="Bandwidth"
              value={analysis.bandwidthHz ? (analysis.bandwidthHz / 1000).toFixed(2) : "N/A"}
              unit="kHz"
            />
            <JarvisStat
              label="Freq Offset"
              value={analysis.freqOffsetHz?.toFixed(2) ?? "N/A"}
              unit="Hz"
            />
            <JarvisStat
              label="I/Q Imbalance"
              value={analysis.iqImbalanceDb?.toFixed(3) ?? "N/A"}
              unit="dB"
            />
            <JarvisStat
              label="DC Offset (Real)"
              value={analysis.dcOffsetReal?.toFixed(4) ?? "N/A"}
            />
            <JarvisStat
              label="DC Offset (Imag)"
              value={analysis.dcOffsetImag?.toFixed(4) ?? "N/A"}
            />
            <div className="flex items-center gap-2">
              {!hasAnomalies ? (
                <>
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-sm text-green-500">No Anomalies</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  <span className="text-sm text-yellow-500">Anomalies Found</span>
                </>
              )}
            </div>
          </div>
        </JarvisPanel>

        {/* Analysis Configuration */}
        {fullMetrics?.analysis_config && (
          <JarvisPanel title="Analysis Configuration">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Sample Rate:</span>
                <span className="ml-2">{(fullMetrics.analysis_config.sample_rate_hz / 1e6).toFixed(2)} MHz</span>
              </div>
              <div>
                <span className="text-muted-foreground">Center Freq:</span>
                <span className="ml-2">{(fullMetrics.analysis_config.center_freq_hz / 1e6).toFixed(2)} MHz</span>
              </div>
              <div>
                <span className="text-muted-foreground">Data Format:</span>
                <span className="ml-2">{fullMetrics.analysis_config.data_format}</span>
              </div>
              <div>
                <span className="text-muted-foreground">FFT Size:</span>
                <span className="ml-2">{fullMetrics.analysis_config.fft_size}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Digital Analysis:</span>
                <span className="ml-2">{fullMetrics.analysis_config.digital_analysis ? "Enabled" : "Disabled"}</span>
              </div>
              <div>
                <span className="text-muted-foreground">V3 Analysis:</span>
                <span className="ml-2">{fullMetrics.analysis_config.v3_analysis ? "Enabled" : "Disabled"}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Analyzer Version:</span>
                <span className="ml-2">{fullMetrics.analyzer_version || "2.2.2"}</span>
              </div>
            </div>
          </JarvisPanel>
        )}
      </div>
    </JarvisLayout>
  );
}
