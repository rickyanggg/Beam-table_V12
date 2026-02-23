
export type ICType = 'TTD' | 'PS';
export type Polarity = 'NORMAL' | 'INVERTED';
export type MappingOrder = 'ROW_MAJOR' | 'COL_MAJOR' | 'CUSTOM';
export type CoordinateSystem = 'AZ_EL' | 'THETA_PHI';

export interface PortMapping {
  chipId: number;
  portId: number;
  elementId?: number; // Logical ID for sorting (e.g., 0 to N-1)
}

export interface PortMappingMap {
  [key: string]: PortMapping; // "x,y" -> {chipId, portId, elementId}
}

export interface ICConfig {
  id: string;
  name: string;
  type: ICType;
  lsb: number; // ps for TTD, deg for PS
  max: number; // ps for TTD, deg for PS
  bits?: number; // Optional: for auto-LSB calculation
  offset: number; // Static hardware offset
  polarity: Polarity; // Direction of logic
}

export type BeamMode = 'TTD_ONLY' | 'PS_ONLY' | 'HYBRID';

export interface HardwareTopology {
  elementsPerChip: number;
  mappingOrder: MappingOrder;
  customMapping?: PortMappingMap;
}

export interface ArrayConfig {
  channelName: string; // 新增：用於識別計算通道
  frequencyGHz: number;
  nx: number;
  ny: number;
  dx_mm: number;
  dy_mm: number;
  mode: BeamMode;
  topology: HardwareTopology;
}

export interface ScanRange {
  system: CoordinateSystem;
  azStart: number;
  azEnd: number;
  azStep: number;
  elStart: number;
  elEnd: number;
  elStep: number;
}

export interface CalibrationMap {
  [key: string]: number; // "x,y" -> value
}

export interface BeamTableRow {
  [key: string]: string | number;
}
