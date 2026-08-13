import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const inputPath = process.argv[2];
const outputPath = process.argv[3];
const inspectPath = process.argv[4];
const renderDir = process.argv[5];

function columnName(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function usedRangeAddress(sheet, rowCount, columnCount) {
  return `A1:${columnName(columnCount - 1)}${rowCount}`;
}

async function inspectWorkbook(workbook, sheets, phase) {
  const records = [];
  for (const sheetSpec of sheets) {
    const sheet = workbook.worksheets.getItem(sheetSpec.name);
    const range = usedRangeAddress(sheet, sheetSpec.rows.length, sheetSpec.rows[0].length);
    const check = await workbook.inspect({
      kind: "table",
      sheetId: sheetSpec.name,
      range,
      include: "values,formulas",
      tableMaxRows: Math.min(sheetSpec.rows.length, 1000),
      tableMaxCols: Math.min(sheetSpec.rows[0].length, 100),
      tableMaxCellChars: 200,
      maxChars: 50000,
    });
    records.push(JSON.stringify({ phase, sheet: sheetSpec.name, range, ndjson: check.ndjson }));
  }
  return records;
}

async function scanFormulaErrors(workbook) {
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
  });
  if (/#(?:REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!)/.test(errors.ndjson || "")) {
    throw new Error(`formula errors detected: ${errors.ndjson.slice(0, 2000)}`);
  }
}

async function main() {
  const request = JSON.parse(await fs.readFile(inputPath, "utf8"));
  const workbook = Workbook.create();
  for (const sheetSpec of request.sheets) {
    const sheet = workbook.worksheets.add(sheetSpec.name);
    const rowCount = sheetSpec.rows.length;
    const columnCount = sheetSpec.rows[0].length;
    const range = sheet.getRange(usedRangeAddress(sheet, rowCount, columnCount));
    range.values = sheetSpec.rows;

    for (const formula of sheetSpec.formulas || []) {
      sheet.getRange(formula.cell).formulas = [[formula.formula]];
      if (formula.number_format) {
        sheet.getRange(formula.cell).format.numberFormat = formula.number_format;
      }
    }

    if (sheetSpec.header_row) {
      sheet.getRange(`A1:${columnName(columnCount - 1)}1`).format = {
        fill: "#1F4E78",
        font: { bold: true, color: "#FFFFFF" },
        wrapText: true,
      };
    }
    const dataStartRow = sheetSpec.header_row ? 2 : 1;
    if (dataStartRow <= rowCount) {
      for (const format of sheetSpec.number_formats || []) {
        sheet.getRange(`${columnName(format.column)}${dataStartRow}:${columnName(format.column)}${rowCount}`).format.numberFormat = format.format;
      }
    }
    for (const [index, width] of (sheetSpec.column_widths || []).entries()) {
      sheet.getRange(`${columnName(index)}:${columnName(index)}`).format.columnWidth = width;
    }
    if (!(sheetSpec.column_widths || []).length) {
      for (let column = 0; column < columnCount; column += 1) {
        const longest = Math.max(...sheetSpec.rows.map((row) => String(row[column] ?? "").length));
        const width = Math.min(40, Math.max(10, longest + 2));
        const columnRange = sheet.getRange(`${columnName(column)}:${columnName(column)}`);
        columnRange.format.columnWidth = width;
        if (longest > 38) columnRange.format.wrapText = true;
      }
      sheet.getUsedRange().format.autofitRows();
    }
    sheet.showGridLines = false;
  }

  const records = await inspectWorkbook(workbook, request.sheets, "created");
  await scanFormulaErrors(workbook);
  await fs.mkdir(renderDir, { recursive: true });
  for (const sheetSpec of request.sheets) {
    const preview = await workbook.render({ sheetName: sheetSpec.name, autoCrop: "all", scale: 1, format: "png" });
    await fs.writeFile(`${renderDir}/${sheetSpec.name.replace(/[^A-Za-z0-9._-]+/g, "_")}.png`, new Uint8Array(await preview.arrayBuffer()));
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  const imported = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
  records.push(...(await inspectWorkbook(imported, request.sheets, "exported")));
  await scanFormulaErrors(imported);
  await fs.writeFile(inspectPath, records.join("\n") + "\n", "utf8");
}

try {
  await main();
  process.exit(0);
} catch (error) {
  process.stderr.write(error instanceof Error ? error.stack || error.message : String(error));
  process.exit(1);
}
