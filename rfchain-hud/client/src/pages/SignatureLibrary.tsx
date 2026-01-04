import { useState } from 'react';
import { trpc } from '@/lib/trpc';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Wifi, Radio, Bluetooth, Satellite, Radar, Tv, 
  Search, Plus, ChevronRight, Signal, Zap 
} from 'lucide-react';

// Category icons mapping
const categoryIcons: Record<string, React.ReactNode> = {
  'WiFi': <Wifi className="h-5 w-5" />,
  'LTE': <Signal className="h-5 w-5" />,
  'Bluetooth': <Bluetooth className="h-5 w-5" />,
  'ZigBee': <Zap className="h-5 w-5" />,
  'LoRa': <Radio className="h-5 w-5" />,
  'Amateur': <Radio className="h-5 w-5" />,
  'Radar': <Radar className="h-5 w-5" />,
  'Broadcast': <Tv className="h-5 w-5" />,
  'Satellite': <Satellite className="h-5 w-5" />,
};

// Built-in signatures data (matches server/signal_signatures.ts)
const BUILT_IN_SIGNATURES = [
  // WiFi
  { name: 'WiFi 802.11b', category: 'WiFi', subcategory: '802.11b', bandwidth: '20-22 MHz', modulation: 'DSSS/CCK', description: 'Legacy 2.4 GHz WiFi' },
  { name: 'WiFi 802.11g', category: 'WiFi', subcategory: '802.11g', bandwidth: '16-20 MHz', modulation: 'OFDM', description: '2.4 GHz WiFi, up to 54 Mbps' },
  { name: 'WiFi 802.11n (HT20)', category: 'WiFi', subcategory: '802.11n', bandwidth: '18-20 MHz', modulation: 'OFDM', description: '2.4/5 GHz with MIMO' },
  { name: 'WiFi 802.11n (HT40)', category: 'WiFi', subcategory: '802.11n', bandwidth: '36-40 MHz', modulation: 'OFDM', description: '40 MHz channel bonding' },
  { name: 'WiFi 802.11ac (VHT80)', category: 'WiFi', subcategory: '802.11ac', bandwidth: '75-80 MHz', modulation: 'OFDM', description: '5 GHz, 80 MHz channel' },
  
  // LTE
  { name: 'LTE 5 MHz', category: 'LTE', subcategory: '5MHz', bandwidth: '4.5-5 MHz', modulation: 'OFDM/SC-FDMA', description: '25 resource blocks' },
  { name: 'LTE 10 MHz', category: 'LTE', subcategory: '10MHz', bandwidth: '9-10 MHz', modulation: 'OFDM/SC-FDMA', description: '50 resource blocks' },
  { name: 'LTE 20 MHz', category: 'LTE', subcategory: '20MHz', bandwidth: '18-20 MHz', modulation: 'OFDM/SC-FDMA', description: '100 resource blocks' },
  
  // Bluetooth
  { name: 'Bluetooth Classic', category: 'Bluetooth', subcategory: 'BR/EDR', bandwidth: '1 MHz', modulation: 'GFSK', description: 'Basic Rate/Enhanced Data Rate' },
  { name: 'Bluetooth Low Energy', category: 'Bluetooth', subcategory: 'BLE', bandwidth: '1-2 MHz', modulation: 'GFSK', description: 'Optimized for IoT' },
  
  // ZigBee
  { name: 'ZigBee 2.4 GHz', category: 'ZigBee', subcategory: '802.15.4', bandwidth: '2-5 MHz', modulation: 'O-QPSK', description: 'IEEE 802.15.4 standard' },
  
  // LoRa
  { name: 'LoRa SF7', category: 'LoRa', subcategory: 'SF7', bandwidth: '125-500 kHz', modulation: 'CSS', description: 'Highest data rate' },
  { name: 'LoRa SF12', category: 'LoRa', subcategory: 'SF12', bandwidth: '125-500 kHz', modulation: 'CSS', description: 'Longest range' },
  
  // Amateur
  { name: 'Amateur SSB Voice', category: 'Amateur', subcategory: 'SSB', bandwidth: '2.4-3 kHz', modulation: 'SSB', description: 'Single Sideband voice' },
  { name: 'Amateur FM Voice', category: 'Amateur', subcategory: 'FM', bandwidth: '10-16 kHz', modulation: 'NBFM', description: 'Narrowband FM' },
  { name: 'Amateur FT8', category: 'Amateur', subcategory: 'FT8', bandwidth: '50 Hz', modulation: '8-GFSK', description: 'Digital weak signal mode' },
  
  // Radar
  { name: 'Pulsed Radar', category: 'Radar', subcategory: 'Pulsed', bandwidth: 'Variable', modulation: 'Pulsed', description: 'Generic pulsed radar' },
  { name: 'FMCW Radar', category: 'Radar', subcategory: 'FMCW', bandwidth: 'Variable', modulation: 'FMCW', description: 'Frequency Modulated CW' },
  
  // Broadcast
  { name: 'DVB-T', category: 'Broadcast', subcategory: 'DVB-T', bandwidth: '6-8 MHz', modulation: 'OFDM', description: 'Digital TV - Terrestrial' },
  { name: 'ATSC', category: 'Broadcast', subcategory: 'ATSC', bandwidth: '5.4-6 MHz', modulation: '8-VSB', description: 'US Digital TV standard' },
];

export default function SignatureLibrary() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  
  // Get unique categories
  const categories = Array.from(new Set(BUILT_IN_SIGNATURES.map(s => s.category)));
  
  // Filter signatures
  const filteredSignatures = BUILT_IN_SIGNATURES.filter(sig => {
    const matchesSearch = searchQuery === '' || 
      sig.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      sig.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      sig.modulation.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesCategory = !selectedCategory || sig.category === selectedCategory;
    
    return matchesSearch && matchesCategory;
  });
  
  // Group by category
  const signaturesByCategory = filteredSignatures.reduce((acc, sig) => {
    if (!acc[sig.category]) acc[sig.category] = [];
    acc[sig.category].push(sig);
    return acc;
  }, {} as Record<string, typeof BUILT_IN_SIGNATURES>);
  
  return (
    <div className="container mx-auto py-8 px-4">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-cyan-400">Signal Signature Library</h1>
          <p className="text-gray-400 mt-1">
            Known signal profiles for automatic classification
          </p>
        </div>
        <Button variant="outline" className="gap-2">
          <Plus className="h-4 w-4" />
          Add Custom Signature
        </Button>
      </div>
      
      {/* Search and Filter */}
      <div className="flex gap-4 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-500" />
          <Input
            placeholder="Search signatures..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-gray-900 border-gray-700"
          />
        </div>
      </div>
      
      {/* Category Tabs */}
      <Tabs defaultValue="all" className="mb-6">
        <TabsList className="bg-gray-900 border border-gray-800">
          <TabsTrigger 
            value="all" 
            onClick={() => setSelectedCategory(null)}
            className="data-[state=active]:bg-cyan-600"
          >
            All ({BUILT_IN_SIGNATURES.length})
          </TabsTrigger>
          {categories.map(cat => (
            <TabsTrigger 
              key={cat} 
              value={cat}
              onClick={() => setSelectedCategory(cat)}
              className="data-[state=active]:bg-cyan-600 gap-2"
            >
              {categoryIcons[cat]}
              {cat}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
      
      {/* Signatures Grid */}
      <div className="space-y-8">
        {Object.entries(signaturesByCategory).map(([category, signatures]) => (
          <div key={category}>
            <div className="flex items-center gap-2 mb-4">
              <span className="text-cyan-400">{categoryIcons[category]}</span>
              <h2 className="text-xl font-semibold text-gray-200">{category}</h2>
              <Badge variant="secondary">{signatures.length}</Badge>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {signatures.map((sig, idx) => (
                <Card 
                  key={`${sig.name}-${idx}`}
                  className="bg-gray-900 border-gray-800 hover:border-cyan-600 transition-colors cursor-pointer"
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg text-gray-200">{sig.name}</CardTitle>
                      <Badge variant="outline" className="text-xs">
                        {sig.subcategory}
                      </Badge>
                    </div>
                    <CardDescription>{sig.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-gray-500">Bandwidth:</span>
                        <p className="text-gray-300">{sig.bandwidth}</p>
                      </div>
                      <div>
                        <span className="text-gray-500">Modulation:</span>
                        <p className="text-gray-300">{sig.modulation}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-800">
                      <Badge className="bg-green-600/20 text-green-400 border-green-600">
                        Built-in
                      </Badge>
                      <Button variant="ghost" size="sm" className="gap-1 text-cyan-400">
                        Details <ChevronRight className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        ))}
      </div>
      
      {filteredSignatures.length === 0 && (
        <div className="text-center py-12">
          <Radio className="h-12 w-12 mx-auto text-gray-600 mb-4" />
          <p className="text-gray-400">No signatures found matching your search.</p>
        </div>
      )}
      
      {/* Info Card */}
      <Card className="mt-8 bg-gray-900 border-cyan-800">
        <CardHeader>
          <CardTitle className="text-cyan-400">How Signature Matching Works</CardTitle>
        </CardHeader>
        <CardContent className="text-gray-300 space-y-2">
          <p>
            When you analyze a signal, the system automatically compares its characteristics 
            against this library of known signal profiles.
          </p>
          <p>
            <strong>Matching criteria include:</strong>
          </p>
          <ul className="list-disc list-inside space-y-1 text-gray-400">
            <li>Bandwidth (weight: 3x)</li>
            <li>Modulation type (weight: 4x)</li>
            <li>Peak-to-Average Power Ratio (weight: 2x)</li>
            <li>OFDM parameters (FFT size, cyclic prefix)</li>
            <li>Symbol rate</li>
          </ul>
          <p className="text-sm text-gray-500 mt-4">
            Matches with confidence above 70% are shown in the analysis results.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
