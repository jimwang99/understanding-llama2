plantuml:
	plantuml model.puml -tpdf -quiet
	plantuml model.puml -tpng -DPLANTUML_LIMIT_SIZE=8192
