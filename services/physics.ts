
import { ICConfig, ArrayConfig, ScanRange, CalibrationMap, BeamTableRow, BeamMode, PortMappingMap } from '../types';

const C = 299792458.0; // m/s

const formatDeg = (val: number): string => {
  return parseFloat(val.toFixed(2)).toString() + 'deg';
};

/**
 * 輔助函數：尋找最接近的標頭索引（不區分大小寫）
 */
const findHeader = (headers: string[], targets: string[]): number => {
  const normalizedHeaders = headers.map(h => h.trim().toLowerCase());
  for (const t of normalizedHeaders) {
    const idx = targets.findIndex(target => target.toLowerCase() === t);
    if (idx !== -1) return normalizedHeaders.indexOf(t);
  }
  return -1;
};

/**
 * 輔助函數：生成掃描範圍點
 */
const generateRange = (start: number, end: number, step: number): number[] => {
  if (step <= 0) return [start];
  const pts: number[] = [];
  
  if (start <= end) {
    for (let v = start; v <= end + 0.0001; v += step) {
      pts.push(v);
    }
  } else {
    const totalDist = (360 - start) + end;
    let distCovered = 0;
    while (distCovered <= totalDist + 0.0001) {
      let val = (start + distCovered) % 360;
      if (val < 0) val += 360;
      pts.push(val);
      distCovered += step;
    }
  }
  return pts;
};

export const getArrayFactorGain = (
  theta: number, 
  phi: number, 
  array: ArrayConfig, 
  betaX: number, 
  betaY: number, 
  k: number
): number => {
  let sumReal = 0;
  let sumImag = 0;

  for (let y = 0; y < array.ny; y++) {
    for (let x = 0; x < array.nx; x++) {
      const x_pos = (x - (array.nx - 1) / 2) * array.dx_mm * 1e-3;
      const y_pos = (y - (array.ny - 1) / 2) * array.dy_mm * 1e-3;

      const phase = k * (x_pos * Math.sin(theta) * Math.cos(phi) + y_pos * Math.sin(theta) * Math.sin(phi)) + 
                    (x_pos * betaX + y_pos * betaY);
      
      sumReal += Math.cos(phase);
      sumImag += Math.sin(phase);
    }
  }

  const af = Math.sqrt(sumReal**2 + sumImag**2) / (array.nx * array.ny);
  const elementFactor = theta <= Math.PI / 2 ? Math.pow(Math.cos(theta), 1.2) : 0;
  const totalField = af * elementFactor;
  return Math.pow(totalField, 1.5);
};

export const calculateBeamTable = (
  array: ArrayConfig,
  scan: ScanRange,
  ics: ICConfig[],
  ttdCal: CalibrationMap = {},
  psCal: CalibrationMap = {}
): BeamTableRow[] => {
  const f_hz = array.frequencyGHz * 1e9;
  const period_ps = (1.0 / f_hz) * 1e12;

  const range1 = generateRange(scan.azStart, scan.azEnd, scan.azStep);
  const range2 = generateRange(scan.elStart, scan.elEnd, scan.elStep);

  const all_data: BeamTableRow[] = [];
  let beamCounter = 1;

  for (const r1Val of range1) {
    for (const r2Val of range2) {
      let targetAz = 0;
      let targetEl = 0;
      let targetTheta = 0;
      let targetPhi = 0;

      if (scan.system === 'THETA_PHI') {
        targetPhi = r1Val;
        targetTheta = r2Val;
        targetAz = targetPhi;
        targetEl = 90 - targetTheta;
      } else {
        targetAz = r1Val;
        targetEl = r2Val;
        targetTheta = 90 - targetEl;
        targetPhi = targetAz;
      }

      const theta_t_rad = (Math.PI / 180) * targetTheta;
      const phi_t_rad = (Math.PI / 180) * targetPhi;

      const temp_grid: any[] = [];
      for (let y = 0; y < array.ny; y++) {
        for (let x = 0; x < array.nx; x++) {
          const x_pos = (x - (array.nx - 1) / 2) * array.dx_mm * 1e-3;
          const y_pos = (y - (array.ny - 1) / 2) * array.dy_mm * 1e-3;
          
          const path_diff = x_pos * Math.sin(theta_t_rad) * Math.cos(phi_t_rad) + 
                          y_pos * Math.sin(theta_t_rad) * Math.sin(phi_t_rad);
          
          const geo_time_sec = -path_diff / C;
          const cal_ttd = ttdCal[`${x},${y}`] || 0.0;
          const cal_ps = psCal[`${x},${y}`] || 0.0;
          
          const { elementsPerChip, mappingOrder, customMapping } = array.topology;
          let chipId = 0;
          let portId = 0;
          let elementId = 0;
          
          const linearIdx = (mappingOrder === 'COL_MAJOR') ? (x * array.ny + y) : (y * array.nx + x);
          elementId = linearIdx;

          if (mappingOrder === 'CUSTOM' && customMapping && customMapping[`${x},${y}`]) {
            chipId = customMapping[`${x},${y}`].chipId;
            portId = customMapping[`${x},${y}`].portId;
            if (customMapping[`${x},${y}`].elementId !== undefined) {
              elementId = customMapping[`${x},${y}`].elementId!;
            }
          } else {
            chipId = Math.floor(linearIdx / elementsPerChip);
            portId = linearIdx % elementsPerChip;
          }

          temp_grid.push({
            x, y, 
            chipId,
            portId,
            elementId,
            raw_sec: geo_time_sec, 
            cal_ttd,
            cal_ps
          });
        }
      }

      const min_sec = Math.min(...temp_grid.map(d => d.raw_sec));
      temp_grid.sort((a, b) => a.elementId - b.elementId);

      for (const item of temp_grid) {
        const theory_delay_ps = (item.raw_sec - min_sec) * 1e12;
        const theory_phase_deg = (theory_delay_ps * 1e-12 * f_hz * 360.0) % 360.0;
        
        const row: BeamTableRow = {
          "Channel": array.channelName, // 新增：輸出通道名稱
          "Beam_ID": beamCounter,
          "Target_Az": Number((targetAz % 360).toFixed(2)),
          "Target_El": Number(targetEl.toFixed(2)),
          "Target_Phi": Number((targetPhi % 360).toFixed(2)),
          "Target_Theta": Number(targetTheta.toFixed(2)),
          "Element_ID": item.elementId,
          "HW_Chip_ID": item.chipId,
          "HW_Port_ID": item.portId,
          "Pos_X": item.x,
          "Pos_Y": item.y,
          "Cal_TTD_ps": item.cal_ttd,
          "Cal_PS_deg": formatDeg(item.cal_ps),
          "Theory_Delay_ps": Number(theory_delay_ps.toFixed(2)),
          "Theory_Phase_deg": formatDeg(theory_phase_deg)
        };

        for (const ic of ics) {
          if (ic.type === 'TTD') {
            let target_logic_ps = 0;
            if (array.mode !== 'PS_ONLY') {
              const combined_input = theory_delay_ps + item.cal_ttd;
              let adjusted = (ic.polarity === 'INVERTED') ? (ic.max - combined_input) : combined_input;
              adjusted += ic.offset;
              
              if (array.mode === 'HYBRID') {
                target_logic_ps = Math.max(0, Math.min(adjusted, ic.max));
              } else {
                target_logic_ps = adjusted % period_ps;
                if (target_logic_ps < 0) target_logic_ps += period_ps;
                if (target_logic_ps > ic.max) target_logic_ps = ic.max;
              }
            }
            const code = Math.round(target_logic_ps / ic.lsb);
            const actual = code * ic.lsb;
            row[`${ic.name}_Actual_ps`] = Number(actual.toFixed(3));
            row[`${ic.name}_Code_Dec`] = code;
            row[`${ic.name}_Code_Hex`] = `0x${code.toString(16).toUpperCase().padStart(2, '0')}`;
            row[`${ic.name}_Error_ps`] = Number((actual - target_logic_ps).toFixed(3));
          } else if (ic.type === 'PS') {
            let target_logic_deg = 0;
            if (array.mode !== 'TTD_ONLY') {
              const combined_input = (theory_phase_deg + item.cal_ps) % 360.0;
              let adjusted = (ic.polarity === 'INVERTED') ? (ic.max - combined_input) : combined_input;
              adjusted += ic.offset;
              target_logic_deg = adjusted % 360.0;
              if (target_logic_deg < 0) target_logic_deg += 360.0;
              if (target_logic_deg > ic.max) target_logic_deg = ic.max;
            }
            const code = Math.round(target_logic_deg / ic.lsb);
            const actual = code * ic.lsb;
            row[`${ic.name}_Actual_deg`] = Number(actual.toFixed(3));
            row[`${ic.name}_Code_Dec`] = code;
            row[`${ic.name}_Code_Hex`] = `0x${code.toString(16).toUpperCase().padStart(2, '0')}`;
            row[`${ic.name}_Error_deg`] = Number((actual - target_logic_deg).toFixed(3));
          }
        }
        all_data.push(row);
      }
      beamCounter++;
    }
  }
  return all_data;
};

export const parseCalibrationFile = async (file: File): Promise<CalibrationMap> => {
  const text = await file.text();
  const lines = text.split('\n');
  const map: CalibrationMap = {};
  if (lines.length < 2) return map;
  
  const headers = lines[0].split(',');
  const xIdx = findHeader(headers, ['x_idx', 'x', 'pos_x']);
  const yIdx = findHeader(headers, ['y_idx', 'y', 'pos_y']);
  const valIdx = findHeader(headers, ['cal_delay_ps', 'cal_phase_deg', 'value', 'cal']);

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',').map(c => c.trim());
    if (cols.length < 3) continue;
    const x = parseInt(cols[xIdx]);
    const y = parseInt(cols[yIdx]);
    const val = parseFloat(cols[valIdx]);
    if (!isNaN(x) && !isNaN(y) && !isNaN(val)) {
      map[`${x},${y}`] = val;
    }
  }
  return map;
};

export const parseMappingFile = async (file: File): Promise<PortMappingMap> => {
  const text = await file.text();
  const lines = text.split('\n');
  const map: PortMappingMap = {};
  if (lines.length < 2) return map;

  const headers = lines[0].split(',');
  const xIdx = findHeader(headers, ['x_idx', 'x', 'pos_x']);
  const yIdx = findHeader(headers, ['y_idx', 'y', 'pos_y']);
  const chipIdx = findHeader(headers, ['chip_id', 'chip', 'hw_chip_id']);
  const portIdx = findHeader(headers, ['port_id', 'port', 'hw_port_id']);
  const eleIdx = findHeader(headers, ['element_id', 'id', 'ant_idx']);

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',').map(c => c.trim());
    if (cols.length < 4) continue;
    const x = parseInt(cols[xIdx]);
    const y = parseInt(cols[yIdx]);
    const cid = parseInt(cols[chipIdx]);
    const pid = parseInt(cols[portIdx]);
    const eid = eleIdx !== -1 ? parseInt(cols[eleIdx]) : undefined;

    if (!isNaN(x) && !isNaN(y) && !isNaN(cid) && !isNaN(pid)) {
      map[`${x},${y}`] = { 
        chipId: cid, 
        portId: pid,
        elementId: (eid !== undefined && !isNaN(eid)) ? eid : undefined
      };
    }
  }
  return map;
};
