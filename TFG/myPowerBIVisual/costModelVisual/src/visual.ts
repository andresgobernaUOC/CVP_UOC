"use strict";

import powerbi from "powerbi-visuals-api";
import * as d3 from "d3";

import VisualConstructorOptions = powerbi.extensibility.visual.VisualConstructorOptions;
import VisualUpdateOptions = powerbi.extensibility.visual.VisualUpdateOptions;
import IVisual = powerbi.extensibility.visual.IVisual;

export class Visual implements IVisual {
    private target: HTMLElement;

    constructor(options: VisualConstructorOptions) {
        this.target = options.element;
    }

    public update(options: VisualUpdateOptions) {
        d3.select(this.target).selectAll("*").remove();

        const width = options.viewport.width;
        const height = options.viewport.height;

        const margin = { top: 40, right: 80, bottom: 80, left: 80 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;

        const svg = d3.select(this.target)
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const dataView = options.dataViews?.[0];
        if (!dataView?.categorical) return;

        const categorical = dataView.categorical;

        // Extract categories
        const cyValues = categorical.categories.find(c => c.source.roles["CY"])?.values as number[] || [];
        const qValues = categorical.categories.find(c => c.source.roles["Q"])?.values as string[] || [];
        const monthValues = categorical.categories.find(c => c.source.roles["month"])?.values as number[] || [];
        const monthOrderValues = categorical.categories.find(c => c.source.roles["monthOrder"])?.values as number[] || [];

        // Extract measures
        const fixCostValues = categorical.values.find(v => v.source.roles["fixCost"])?.values as number[] || [];
        const varCostValues = categorical.values.find(v => v.source.roles["varCost"])?.values as number[] || [];
        const costDriverValues = categorical.values.find(v => v.source.roles["costDriverInput"])?.values as number[] || [];
        const varUnitCostValues = categorical.values.find(v => v.source.roles["modVarUnitCost"])?.values as number[] || [];

        // Median helper function
        function computeMedian(values: number[]): number {
            if (!values.length) return 0;
            const sorted = values.slice().sort((a, b) => a - b);
            const mid = Math.floor(sorted.length / 2);
            if (sorted.length % 2 === 0) {
                return (sorted[mid - 1] + sorted[mid]) / 2;
            } else {
                return sorted[mid];
            }
        }

        // Extract single measure values correctly
        function extractSingleMeasure(roleName: string): number {
            const col = categorical.values.find(v => v.source.roles[roleName]);
            if (!col || !col.values || col.values.length === 0) return 0;

            const vals = col.values as number[];

            switch (roleName) {
                case "fixCostMin":
                    return d3.min(vals) || 0;
                case "fixCostMax":
                    return d3.max(vals) || 0;
                case "fixCostMedian":
                    return computeMedian(vals);
                default:
                    return vals.length > 0 ? vals[0] : 0;
            }
        }

        const fixCostMedian = extractSingleMeasure("fixCostMedian");
        const fixCostMin = extractSingleMeasure("fixCostMin");
        const fixCostMax = extractSingleMeasure("fixCostMax");

        if (!cyValues.length || !qValues.length || !monthValues.length || !monthOrderValues.length ||
            !fixCostValues.length || !varCostValues.length) {
            return;
        }

        // Compose label and data array
        let data = cyValues.map((cy, i) => ({
            category: `${cy}-${qValues[i]}-${monthValues[i]}`,
            fixCost: fixCostValues[i] || 0,
            varCost: varCostValues[i] || 0,
            monthOrder: monthOrderValues[i] || 0,
            costDriver: costDriverValues[i] || 0,
            varUnitCost: varUnitCostValues[i] || 0
        }));

        // Sort by monthOrder ascending
        data = data.sort((a, b) => a.monthOrder - b.monthOrder);

        // X scale domain from sorted data
        const xDomain = data.map(d => d.category);

        const x = d3.scalePoint()
            .domain(xDomain)
            .range([0, innerWidth])
            .padding(0.5);

        // Y scale for stacked costs
        const maxYValue = d3.max([
            d3.max(data, d => d.fixCost + d.varCost) || 0,
            fixCostMedian,
            fixCostMin,
            fixCostMax
        ]) || 0;

        const yLeft = d3.scaleLinear()
            .domain([0, maxYValue])
            .nice()
            .range([innerHeight, 0]);

        // Y scale for costDriver (volume) on secondary axis
        const maxCostDriver = d3.max(data, d => d.costDriver) || 0;
        const yRight = d3.scaleLinear()
            .domain([0, maxCostDriver])
            .nice()
            .range([innerHeight, 0]);

        // Y scale for varUnitCost (third axis)
        const maxVarUnitCost = d3.max(data, d => d.varUnitCost) || 0;
        const yRightVarUnit = d3.scaleLinear()
            .domain([0, maxVarUnitCost])
            .nice()
            .range([innerHeight, 0]);

        // D3 stack generator for fixCost and varCost
        const stack = d3.stack()
            .keys(["fixCost", "varCost"]);

        const series = stack(data as any);

        // Area generator for stacked areas
        const area = d3.area<[number, number] & { data: any }>()
            .x(d => {
                const category = d.data.category;
                const xVal = x(category);
                return xVal !== undefined ? xVal : 0;
            })
            .y0(d => yLeft(d[0]))
            .y1(d => yLeft(d[1]))
            .curve(d3.curveMonotoneX);

        // Colors for stacked areas
        const colors = ["#6B007B", "#118DFF"];

        svg.selectAll("path")
            .data(series)
            .enter()
            .append("path")
            .attr("fill", (_, i) => colors[i])
            .attr("d", d => area(d as any));

        // Draw horizontal dashed lines with white color and labels (left axis)
        function drawLine(value: number, label: string) {
            const opacity = value > 0 ? 1 : 0; // Hide line and label if value is 0 or not populated

            svg.append("line")
                .attr("x1", 0)
                .attr("x2", innerWidth)
                .attr("y1", yLeft(value))
                .attr("y2", yLeft(value))
                .attr("stroke", "white")
                .attr("stroke-width", 2)
                .style("stroke-dasharray", "4 2")
                .style("opacity", opacity);

            svg.append("text")
                .attr("x", innerWidth - 5)
                .attr("y", yLeft(value) - 5)
                .attr("text-anchor", "end")
                .style("font-size", "9px")
                .style("fill", "white")
                .style("opacity", opacity)
                .text(label);
        }

        drawLine(fixCostMedian, "Fix Cost Median (modeled)");
        drawLine(fixCostMin, "Min Fix Cost (modeled)");
        drawLine(fixCostMax, "Max Fix Cost (modeled)");

        // Volume line color
        const volumeLineColor = "#E66C37";

        // Draw costDriverInput line (secondary axis, right)
        const lineCostDriver = d3.line<any>()
            .x(d => {
                const xVal = x(d.category);
                return xVal !== undefined ? xVal : 0;
            })
            .y(d => yRight(d.costDriver))
            .curve(d3.curveMonotoneX);

        svg.append("path")
            .datum(data)
            .attr("fill", "none")
            .attr("stroke", volumeLineColor)
            .attr("stroke-width", 2)
            .attr("d", lineCostDriver);

        // Variable Unit Cost line color and width with white border effect
        const varUnitCostLineColor = "#000000";

        // Path generator for varUnitCost line
        const lineVarUnitCost = d3.line<any>()
            .x(d => {
                const xVal = x(d.category);
                return xVal !== undefined ? xVal : 0;
            })
            .y(d => yRightVarUnit(d.varUnitCost))
            .curve(d3.curveMonotoneX);

        // Draw white "border" line underneath (thicker)
        svg.append("path")
            .datum(data)
            .attr("fill", "none")
            .attr("stroke", "white")
            .attr("stroke-width", 5)
            .style("stroke-dasharray", "5,5")
            .attr("d", lineVarUnitCost);

        // Draw black line on top (thinner)
        svg.append("path")
            .datum(data)
            .attr("fill", "none")
            .attr("stroke", varUnitCostLineColor)
            .attr("stroke-width", 4)
            .style("stroke-dasharray", "5,5")
            .attr("d", lineVarUnitCost);

        // X axis with rotated labels
        svg.append("g")
            .attr("transform", `translate(0,${innerHeight})`)
            .call(d3.axisBottom(x))
            .selectAll("text")
            .attr("transform", "rotate(-45)")
            .style("text-anchor", "end")
            .style("font-size", "9px")
            .style("fill", "white");

        // Left Y axis for costs
        svg.append("g")
            .call(d3.axisLeft(yLeft))
            .selectAll("text")
            .style("font-size", "9px")
            .style("fill", "white");

        // Right Y axis for volume (costDriver)
        svg.append("g")
            .attr("transform", `translate(${innerWidth},0)`)
            .call(d3.axisRight(yRight))
            .selectAll("text")
            .style("font-size", "9px")
            .style("fill", "white");

        // Second right Y axis for varUnitCost - moved 10px further right
        svg.append("g")
            .attr("transform", `translate(${innerWidth + 50},0)`) // increased from 40 to 50
            .call(d3.axisRight(yRightVarUnit))
            .selectAll("text")
            .style("font-size", "9px")
            .style("fill", "white");

        // Axis titles
        svg.append("text")
            .attr("transform", `translate(${innerWidth / 2},${innerHeight + margin.bottom + 20})`) // moved below axis ticks
            .style("text-anchor", "middle")
            .style("font-size", "9px")
            .style("fill", "white")
            .text("Period (CY-Q-Month)");

        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", -margin.left + 30)  // moved above ticks
            .attr("x", -innerHeight / 2)
            .style("text-anchor", "middle")
            .style("font-size", "9px")
            .style("fill", "white")
            .text("Cost ($)");

        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", innerWidth + margin.right + 15)  // above ticks
            .attr("x", -innerHeight / 2)
            .style("text-anchor", "middle")
            .style("font-size", "9px")
            .style("fill", "white")
            .text("Volume");

        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", innerWidth + margin.right + 85) // moved from 75 to 85 to match axis shift
            .attr("x", -innerHeight / 2)
            .style("text-anchor", "middle")
            .style("font-size", "9px")
            .style("fill", "white")
            .text("Variable Unit Cost");

        // Legend for stacked areas + volume line + varUnitCost line
        const legendData = [
            { label: "Fixed Cost", color: colors[0] },
            { label: "Variable Cost", color: colors[1] },
            { label: "Volume", color: volumeLineColor },
            { label: "Variable Unit Cost", color: varUnitCostLineColor }
        ];

        const legend = svg.append("g")
            .attr("transform", `translate(0,${height - margin.bottom + 20})`);

        legend.selectAll("rect")
            .data(legendData)
            .enter()
            .append("rect")
            .attr("x", (_, i) => i * 150)
            .attr("width", 14)
            .attr("height", 14)
            .attr("fill", d => d.color)
            .attr("stroke", "white")
            .attr("stroke-width", d => d.color === "#000000" ? 1 : 0);

        legend.selectAll("text")
            .data(legendData)
            .enter()
            .append("text")
            .attr("x", (_, i) => i * 150 + 20)
            .attr("y", 12)
            .text(d => d.label)
            .style("font-size", "9px")
            .style("fill", "white");
    }
}
