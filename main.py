import os
import zipfile
import json
import re
import logging
import tempfile
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, List, Any, Optional
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open("config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)
except FileNotFoundError:
    logger.warning("Config file not found. Using default configuration.")
    config = {
        "extraction": {
            "temp_dir": "/tmp/pbix_extract",
            "max_file_size_mb": 100
        },
        "api": {
            "host": "",
            "port": 8000
        },
        "insights": {
            "max_recommendations": 5,
            "analyze_performance": True,
            "analyze_data_model": True,
            "analyze_visuals": True
        }
    }

# Create FastAPI app
app = FastAPI(title="PBIX Insights Generator",
              description="API for parsing Power BI files and generating insights",
              version="1.0.0")


class PBIXParser:
    """
    Class for extracting and parsing Power BI (.pbix) files
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.temp_dir = config["extraction"]["temp_dir"]
        self.extraction_path = tempfile.mkdtemp(dir=self.temp_dir)
        self.layout_data = None
        self.data_model_info = None
        self.query_info = None

    def extract_pbix(self) -> bool:
        """Extract the PBIX file which is essentially a zip archive"""
        try:
            with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
                zip_ref.extractall(self.extraction_path)
            logger.info(
                f"Successfully extracted PBIX to {self.extraction_path}")
            return True
        except zipfile.BadZipFile:
            logger.error("The file is not a valid PBIX (zip) file")
            return False
        except Exception as e:
            logger.error(f"Error extracting PBIX: {str(e)}")
            return False

    def parse_layout(self) -> Dict:
        """Extract report layout information"""
        try:
            layout_path = os.path.join(
                self.extraction_path, 'Report', 'Layout')
            if not os.path.exists(layout_path):
                logger.warning("Layout file not found")
                return {}

            with open(layout_path, 'r', encoding='utf-16-le') as f:
                content = f.read()

            # Remove BOM if present
            if content.startswith('\ufeff'):
                content = content[1:]

            self.layout_data = json.loads(content)

            # Extract visual information
            visuals = []
            if 'sections' in self.layout_data:
                for section in self.layout_data['sections']:
                    if 'visualContainers' in section:
                        for container in section['visualContainers']:
                            if 'config' in container:
                                visual_config = json.loads(container['config'])
                                visual_type = visual_config.get(
                                    'singleVisual', {}).get('visualType', 'unknown')
                                visuals.append({
                                    'type': visual_type,
                                    'config': visual_config
                                })

            return {
                'visuals_count': len(visuals),
                'visual_types': list(set(v['type'] for v in visuals)),
                'visuals': visuals
            }
        except Exception as e:
            logger.error(f"Error parsing layout: {str(e)}")
            return {}

    def parse_data_model(self) -> Dict:
        """Extract and parse data model information"""
        try:
            data_model_path = os.path.join(self.extraction_path, 'DataModel')
            if not os.path.exists(data_model_path):
                logger.warning("Data model file not found")
                return {}

            # In a real implementation, this would parse the complex binary format
            # For this example, we'll return simplified information

            # Look for DataModelSchema file
            schema_path = os.path.join(self.extraction_path, 'DataModelSchema')
            tables = []
            relationships = []

            if os.path.exists(schema_path):
                with open(schema_path, 'r', encoding='utf-8') as f:
                    try:
                        schema_data = json.loads(f.read())

                        # Extract table information
                        if 'model' in schema_data and 'tables' in schema_data['model']:
                            for table in schema_data['model']['tables']:
                                table_info = {
                                    'name': table.get('name', 'Unknown'),
                                    'columns': []
                                }

                                if 'columns' in table:
                                    for column in table['columns']:
                                        table_info['columns'].append({
                                            'name': column.get('name', 'Unknown'),
                                            'dataType': column.get('dataType', 'Unknown')
                                        })

                                tables.append(table_info)

                        # Extract relationships
                        if 'model' in schema_data and 'relationships' in schema_data['model']:
                            for rel in schema_data['model']['relationships']:
                                relationships.append({
                                    'fromTable': rel.get('fromTable', 'Unknown'),
                                    'fromColumn': rel.get('fromColumn', 'Unknown'),
                                    'toTable': rel.get('toTable', 'Unknown'),
                                    'toColumn': rel.get('toColumn', 'Unknown'),
                                    'cardinality': rel.get('cardinality', 'Unknown')
                                })
                    except json.JSONDecodeError:
                        logger.warning(
                            "Could not parse DataModelSchema as JSON")

            self.data_model_info = {
                'tables': tables,
                'tables_count': len(tables),
                'relationships': relationships,
                'relationships_count': len(relationships)
            }

            return self.data_model_info
        except Exception as e:
            logger.error(f"Error parsing data model: {str(e)}")
            return {}

    def parse_queries(self) -> Dict:
        """Extract and parse Power Query information"""
        try:
            section_path = os.path.join(
                self.extraction_path, 'Metadata', 'Section')
            if not os.path.exists(section_path):
                logger.warning("Section file (containing queries) not found")
                return {}

            # Read the content of Section file
            with open(section_path, 'r', encoding='utf-8') as f:
                section_content = f.read()

            # Try to extract M queries
            queries = []
            query_matches = re.finditer(
                r'"Query"\s*:\s*"(.*?)(?<!\\)"(?=,|\s*})', section_content, re.DOTALL)

            for match in query_matches:
                # Unescape any escaped quotes in the query
                query_text = match.group(1).replace('\\"', '"')
                queries.append(query_text)

            self.query_info = {
                'queries_count': len(queries),
                'queries': queries
            }

            return self.query_info
        except Exception as e:
            logger.error(f"Error parsing queries: {str(e)}")
            return {}

    def cleanup(self) -> None:
        """Remove temporary extraction directory"""
        try:
            if os.path.exists(self.extraction_path):
                shutil.rmtree(self.extraction_path)
            logger.info(
                f"Cleaned up extraction directory: {self.extraction_path}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def parse_all(self) -> Dict:
        """Extract and parse all components of the PBIX file"""
        if not self.extract_pbix():
            return {"error": "Failed to extract PBIX file"}

        result = {
            "layout": self.parse_layout(),
            "data_model": self.parse_data_model(),
            "queries": self.parse_queries()
        }

        self.cleanup()
        return result


class InsightsGenerator:
    """
    Class for generating insights based on PBIX file content
    """

    def __init__(self, parsed_data: Dict):
        self.parsed_data = parsed_data
        self.insights = []
        self.max_recommendations = config["insights"]["max_recommendations"]

    def analyze_data_model(self) -> List[Dict]:
        """Analyze data model to generate insights"""
        insights = []

        if not self.parsed_data.get('data_model'):
            return []

        data_model = self.parsed_data['data_model']

        # Check for missing relationships
        if data_model.get('tables_count', 0) > 1 and data_model.get('relationships_count', 0) == 0:
            insights.append({
                'type': 'warning',
                'category': 'data_model',
                'title': 'Missing relationships between tables',
                'description': 'Your model contains multiple tables but no relationships. Consider adding appropriate relationships between tables.',
                'impact': 'high'
            })

        # Check for table without any measures
        measure_count = 0
        for table in data_model.get('tables', []):
            for column in table.get('columns', []):
                if column.get('dataType') == 'calculated':
                    measure_count += 1

        if measure_count == 0 and data_model.get('tables_count', 0) > 0:
            insights.append({
                'type': 'suggestion',
                'category': 'data_model',
                'title': 'No measures defined',
                'description': 'Consider adding calculated measures to improve analysis capabilities.',
                'impact': 'medium'
            })

        # Check for star schema compliance
        tables_with_relationships = set()
        fact_table_candidates = set()

        for rel in data_model.get('relationships', []):
            tables_with_relationships.add(rel.get('fromTable'))
            tables_with_relationships.add(rel.get('toTable'))

            # Tables that connect to multiple other tables are fact table candidates
            fact_table_candidates.add(rel.get('fromTable'))

        if len(tables_with_relationships) > 3 and len(fact_table_candidates) == 0:
            insights.append({
                'type': 'suggestion',
                'category': 'data_model',
                'title': 'Star schema not detected',
                'description': 'Your data model could benefit from a star schema design with central fact tables connected to dimension tables.',
                'impact': 'medium'
            })

        return insights

    def analyze_visuals(self) -> List[Dict]:
        """Analyze report visuals to generate insights"""
        insights = []

        if not self.parsed_data.get('layout'):
            return []

        layout = self.parsed_data['layout']
        visual_types = layout.get('visual_types', [])
        visuals_count = layout.get('visuals_count', 0)

        # Check for overuse of pie charts
        pie_charts = sum(
            1 for v_type in visual_types if 'pie' in v_type.lower())
        if pie_charts > 2:
            insights.append({
                'type': 'suggestion',
                'category': 'visuals',
                'title': 'Overuse of pie charts',
                'description': 'Your report contains multiple pie charts. Consider using bar charts for better data comparison.',
                'impact': 'medium'
            })

        # Check for lack of visual variety
        if len(visual_types) < 3 and visuals_count > 5:
            insights.append({
                'type': 'suggestion',
                'category': 'visuals',
                'title': 'Limited visual variety',
                'description': 'Your report uses a limited set of visual types. Consider diversifying with appropriate visualization types.',
                'impact': 'low'
            })

        # Check for excessive visuals on one page
        # In a simplified model, we can't detect pages easily, so we'll approximate
        if visuals_count > 10:
            insights.append({
                'type': 'warning',
                'category': 'visuals',
                'title': 'Too many visuals',
                'description': 'Your report may contain too many visuals. Consider breaking into multiple pages or focusing on key metrics.',
                'impact': 'medium'
            })

        # Check for lack of filters or slicers
        has_slicers = any('slicer' in v_type.lower()
                          for v_type in visual_types)
        if not has_slicers and visuals_count > 3:
            insights.append({
                'type': 'suggestion',
                'category': 'visuals',
                'title': 'No slicers detected',
                'description': 'Your report doesn\'t include any slicers. Consider adding slicers or filters for better data exploration.',
                'impact': 'medium'
            })

        return insights

    def analyze_performance(self) -> List[Dict]:
        """Analyze performance aspects of the PBIX file"""
        insights = []

        # Check for complex queries
        if self.parsed_data.get('queries', {}).get('queries_count', 0) > 0:
            queries = self.parsed_data.get('queries', {}).get('queries', [])
            complex_queries = []

            for idx, query in enumerate(queries):
                # Simple heuristics for query complexity
                complexity_score = 0
                if len(query) > 500:
                    complexity_score += 1
                if query.count('Table.Join') > 2:
                    complexity_score += 1
                if query.count('Table.NestedJoin') > 0:
                    complexity_score += 2
                if query.count('Table.Group') > 1:
                    complexity_score += 1

                if complexity_score >= 2:
                    complex_queries.append(idx)

            if len(complex_queries) > 0:
                insights.append({
                    'type': 'warning',
                    'category': 'performance',
                    'title': 'Complex Power Query transformations',
                    'description': f'Found {len(complex_queries)} queries with complex transformations that might impact performance.',
                    'impact': 'high',
                    'details': f'Query indices with high complexity: {complex_queries}'
                })

        # Check data model size estimation
        tables_count = self.parsed_data.get(
            'data_model', {}).get('tables_count', 0)
        total_columns = 0

        for table in self.parsed_data.get('data_model', {}).get('tables', []):
            total_columns += len(table.get('columns', []))

        if tables_count > 15 or total_columns > 100:
            insights.append({
                'type': 'warning',
                'category': 'performance',
                'title': 'Large data model',
                'description': f'Your data model contains {tables_count} tables and {total_columns} columns which could impact performance.',
                'impact': 'medium',
                'recommendation': 'Consider simplifying your data model by removing unused tables/columns or using aggregations.'
            })

        return insights

    def generate_all_insights(self) -> Dict:
        """Generate all insights based on the parsed PBIX content"""
        all_insights = []

        if config["insights"]["analyze_data_model"]:
            all_insights.extend(self.analyze_data_model())

        if config["insights"]["analyze_visuals"]:
            all_insights.extend(self.analyze_visuals())

        if config["insights"]["analyze_performance"]:
            all_insights.extend(self.analyze_performance())

        # Sort insights by impact
        impact_scores = {'high': 3, 'medium': 2, 'low': 1}
        all_insights.sort(key=lambda x: impact_scores.get(
            x.get('impact', 'low'), 0), reverse=True)

        # Limit to max recommendations
        all_insights = all_insights[:self.max_recommendations]

        # Add summary statistics
        summary = {
            'total_insights': len(all_insights),
            'by_category': {},
            'by_impact': {}
        }

        for insight in all_insights:
            category = insight.get('category', 'other')
            impact = insight.get('impact', 'low')

            if category not in summary['by_category']:
                summary['by_category'][category] = 0
            summary['by_category'][category] += 1

            if impact not in summary['by_impact']:
                summary['by_impact'][impact] = 0
            summary['by_impact'][impact] += 1

        return {
            'insights': all_insights,
            'summary': summary
        }


# API endpoints
@app.post("/api/analyze")
async def analyze_pbix(file: UploadFile = File(...)):
    """
    Analyze a PBIX file and return insights
    """
    # Check file extension
    if not file.filename.lower().endswith('.pbix'):
        raise HTTPException(
            status_code=400, detail="Only .pbix files are supported")

    # Check file size
    max_size_bytes = config["extraction"]["max_file_size_mb"] * 1024 * 1024
    file_size = 0

    # Ensure temp directory exists
    os.makedirs(config["extraction"]["temp_dir"], exist_ok=True)

    try:
        # Save uploaded file to temp location
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix='.pbix', dir=config["extraction"]["temp_dir"])

        # Read and write file in chunks to avoid memory issues with large files
        chunk_size = 1024 * 1024  # 1MB chunks
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            file_size += len(chunk)
            if file_size > max_size_bytes:
                temp_file.close()
                os.unlink(temp_file.name)
                raise HTTPException(
                    status_code=400, detail=f"File too large. Maximum allowed size is {config['extraction']['max_file_size_mb']} MB")
            temp_file.write(chunk)

        temp_file.close()

        # Parse PBIX file
        parser = PBIXParser(temp_file.name)
        parsed_data = parser.parse_all()

        # Generate insights from the parsed data
        insights_generator = InsightsGenerator(parsed_data)
        insights = insights_generator.generate_all_insights()

        # Combine parsed data and insights
        result = {
            "file_info": {
                "name": file.filename,
                "size_bytes": file_size
            },
            "parsed_data": parsed_data,
            "insights": insights
        }

        # Clean up the temporary file
        os.unlink(temp_file.name)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing PBIX file: {str(e)}")


@app.get("/api/status")
async def get_status():
    """Get API status"""
    return {"status": "operational", "version": "1.0.0"}


if __name__ == "__main__":
    # Create temp directory if it doesn't exist
    os.makedirs(config["extraction"]["temp_dir"], exist_ok=True)

    # Start the API server
    uvicorn.run(
        "main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True
    )
