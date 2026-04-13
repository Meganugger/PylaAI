import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    id: root
    visible: true
    width: 1520
    height: 900
    minimumWidth: 1180
    minimumHeight: 760
    color: "#090A0E"
    title: state.versionTag || "PylaAI"

    property var state: ({})
    property var roster: []
    property var brawlers: []
    property var history: []
    property var live: ({})
    property int pageIndex: 0
    property string selectedBrawler: ""
    property string toastText: ""
    property string toastLevel: "info"

    readonly property color bg: "#090A0E"
    readonly property color sidebar: "#07080C"
    readonly property color panel: "#11141B"
    readonly property color panelAlt: "#171C25"
    readonly property color panelHover: "#1C2330"
    readonly property color fieldFill: "#0D1016"
    readonly property color border: "#2C3442"
    readonly property color textMain: "#F4F7FB"
    readonly property color textDim: "#9DA7B8"
    readonly property color accent: "#ED2A2A"
    readonly property color accentSoft: "#5D1E23"
    readonly property color success: "#4FD08E"
    readonly property color warning: "#F5BC52"
    readonly property color danger: "#F16C6C"
    readonly property color info: "#72B7FF"
    readonly property color gold: "#FFD166"
    readonly property int outerGap: 22
    readonly property int cardGap: 16
    readonly property int cardRadius: 20
    readonly property int fieldHeight: 48

    function notify(level, message) { toastLevel = level; toastText = message; toast.open() }
    function colorForLevel(level) { return level === "success" ? success : level === "error" ? danger : level === "warning" ? warning : info }
    function yesNo(v) { return v ? "yes" : "no" }
    function boolFrom(v) { return String(v).toLowerCase() === "yes" || v === true }
    function selectedData() {
        for (let i = 0; i < brawlers.length; ++i) if (brawlers[i].name === selectedBrawler) return brawlers[i]
        return ({})
    }
    function rosterEntry() {
        for (let i = 0; i < roster.length; ++i) if (roster[i].brawler === selectedBrawler) return roster[i]
        return null
    }
    function secondsSinceStart() { return live.start_time ? Math.max(0, Math.floor(Date.now() / 1000 - Number(live.start_time))) : 0 }
    function formatDuration(seconds) {
        const hrs = Math.floor(seconds / 3600)
        const mins = Math.floor((seconds % 3600) / 60)
        const secs = seconds % 60
        function pad(v) { return v < 10 ? "0" + v : "" + v }
        return pad(hrs) + ":" + pad(mins) + ":" + pad(secs)
    }
    function liveMetricNumber(value, fallback) {
        const n = Number(value)
        return isNaN(n) ? fallback : n
    }
    function comboItemText(model, index, role) {
        if (!model)
            return ""
        let item = undefined
        if (model.count !== undefined && model.get) {
            if (index < 0 || index >= model.count)
                return ""
            item = model.get(index)
        } else {
            if (index < 0 || index >= model.length)
                return ""
            item = model[index]
        }
        if (item === undefined || item === null)
            return ""
        if (role && typeof item === "object" && item[role] !== undefined)
            return String(item[role])
        return typeof item === "object" ? String(item.text || item.label || "") : String(item)
    }
    function checkedCount(listModel) {
        let count = 0
        for (let i = 0; i < listModel.count; ++i) {
            if (listModel.get(i).checked)
                count += 1
        }
        return count
    }
    function hasCapability(name) {
        return !!(state.capabilities && state.capabilities[name])
    }
    function rebuildComboModels() {
        gamemodeModel.clear()
        emulatorModel.clear()
        const incomingModes = state.gamemodes || []
        for (let i = 0; i < incomingModes.length; ++i) {
            const item = incomingModes[i]
            gamemodeModel.append({
                "value": String(item && item.value !== undefined ? item.value : ""),
                "label": String(item && item.label !== undefined ? item.label : "")
            })
        }
        const incomingEmulators = state.emulators || []
        for (let j = 0; j < incomingEmulators.length; ++j)
            emulatorModel.append({ "label": String(incomingEmulators[j]) })
    }
    function hydrate(newState) {
        state = newState || {}
        roster = (state.roster || []).slice()
        brawlers = (state.brawlers || []).slice()
        history = (state.history || []).slice()
        live = state.live || {}
        rebuildComboModels()
        if (!selectedBrawler && brawlers.length) selectedBrawler = brawlers[0].name
        hydrateEditors()
    }
    function hydrateEditors() {
        let d = rosterEntry() || selectedData()
        if (!d || (!d.name && !d.brawler)) return
        brawlerTitle.text = d.displayName || d.brawler || ""
        trophiesField.text = String(d.trophies || 0)
        winsField.text = String(d.wins || 0)
        targetField.text = String(d.push_until || d.pushUntil || state.general.auto_push_target_trophies || 1000)
        streakField.text = String(d.win_streak || d.winStreak || 0)
        typeBox.currentIndex = Math.max(0, ["trophies","wins","quest"].indexOf(d.type || "trophies"))
        autoPick.checked = d.automatically_pick === undefined ? true : !!d.automatically_pick
        manualTrophies.checked = !!d.manual_trophies
    }
    function saveControl() {
        let gamemodeValue = "knockout"
        if (modeBox.currentIndex >= 0 && modeBox.currentIndex < gamemodeModel.count)
            gamemodeValue = gamemodeModel.get(modeBox.currentIndex).value
        backend.saveControlSettings({"map_orientation": orientationBox.currentText.toLowerCase(), "current_emulator": emulatorBox.currentText, "run_for_minutes": timerField.text, "gamemode": gamemodeValue})
    }
    function saveBrawler() {
        backend.addOrUpdateRosterEntry({"brawler": selectedBrawler, "trophies": trophiesField.text, "wins": winsField.text, "push_until": targetField.text, "type": typeBox.currentText.toLowerCase(), "automatically_pick": autoPick.checked, "win_streak": streakField.text, "manual_trophies": manualTrophies.checked})
    }
    function saveFarm() {
        let excluded = []
        for (let i = 0; i < excludeModel.count; ++i) if (excludeModel.get(i).checked) excluded.push(excludeModel.get(i).name)
        let questExcluded = []
        for (let j = 0; j < questExcludeModel.count; ++j) if (questExcludeModel.get(j).checked) questExcluded.push(questExcludeModel.get(j).name)
        backend.saveFarmSettings({"smart_trophy_farm": farmEnabled.checked, "trophy_farm_target": farmTarget.text, "trophy_farm_strategy": farmStrategy.currentText, "trophy_farm_excluded": excluded, "quest_farm_enabled": questEnabled.checked, "quest_farm_mode": questMode.currentText, "quest_farm_excluded": questExcluded})
    }
    function saveSettings() {
        backend.saveSettings({
            "general": {"max_ips": maxIps.text, "cpu_or_gpu": backendBox.currentText, "super_debug": yesNo(debugBox.checked), "personal_webhook": webhookField.text, "discord_id": discordField.text, "brawlstars_api_key": bsApiField.text, "brawlstars_player_tag": playerTagField.text, "api_base_url": apiBaseField.text, "brawlstars_package": packageField.text, "emulator_port": portField.text, "run_for_minutes": settingsTimer.text, "auto_push_target_trophies": autoPushField.text, "current_emulator": settingsEmulator.currentText, "map_orientation": settingsOrientation.currentText.toLowerCase()},
            "bot": {"minimum_movement_delay": minMove.text, "unstuck_movement_delay": unstuckDelay.text, "unstuck_movement_hold_time": unstuckHold.text, "wall_detection_confidence": wallConf.text, "entity_detection_confidence": entityConf.text, "seconds_to_hold_attack_after_reaching_max": holdAttack.text, "play_again_on_win": yesNo(playAgain.checked), "bot_uses_gadgets": yesNo(useGadgets.checked)},
            "time": {"state_check": stateCheck.text, "no_detections": noDetect.text, "idle": idleField.text, "gadget": gadgetField.text, "hypercharge": hyperField.text, "super": superField.text, "wall_detection": wallField.text, "no_detection_proceed": noProceed.text, "check_if_brawl_stars_crashed": crashCheck.text},
            "login": {"key": pylaKey.text}
        })
    }
    function applyStateToForms() {
        if (!state.general) return
        orientationBox.currentIndex = Math.max(0, ["vertical","horizontal"].indexOf(String(state.general.map_orientation || "vertical").toLowerCase()))
        settingsOrientation.currentIndex = orientationBox.currentIndex
        timerField.text = String(state.general.run_for_minutes || 600)
        settingsTimer.text = timerField.text
        let emuIndex = 0
        for (let emu = 0; emu < emulatorModel.count; ++emu) if (emulatorModel.get(emu).label === (state.general.current_emulator || "LDPlayer")) emuIndex = emu
        emulatorBox.currentIndex = Math.max(0, emuIndex)
        settingsEmulator.currentIndex = Math.max(0, emuIndex)
        let gm = String(state.bot.gamemode || "knockout")
        let idx = 0
        for (let i = 0; i < gamemodeModel.count; ++i) if (gamemodeModel.get(i).value === gm) idx = i
        modeBox.currentIndex = idx
        maxIps.text = String(state.general.max_ips || "auto")
        backendBox.currentIndex = Math.max(0, ["auto","cpu","gpu"].indexOf(String(state.general.cpu_or_gpu || "auto").toLowerCase()))
        debugBox.checked = boolFrom(state.general.super_debug)
        webhookField.text = String(state.general.personal_webhook || "")
        discordField.text = String(state.general.discord_id || "")
        bsApiField.text = String(state.general.brawlstars_api_key || "")
        playerTagField.text = String(state.general.brawlstars_player_tag || "")
        apiBaseField.text = String(state.general.api_base_url || "localhost")
        packageField.text = String(state.general.brawlstars_package || "com.supercell.brawlstars")
        portField.text = String(state.general.emulator_port || 5037)
        autoPushField.text = String(state.general.auto_push_target_trophies || 1000)
        pylaKey.text = String((state.login || {}).key || "")
        minMove.text = String(state.bot.minimum_movement_delay || 0.08)
        unstuckDelay.text = String(state.bot.unstuck_movement_delay || 1.5)
        unstuckHold.text = String(state.bot.unstuck_movement_hold_time || 0.8)
        wallConf.text = String(state.bot.wall_detection_confidence || 0.9)
        entityConf.text = String(state.bot.entity_detection_confidence || 0.6)
        holdAttack.text = String(state.bot.seconds_to_hold_attack_after_reaching_max || 1.5)
        playAgain.checked = boolFrom(state.bot.play_again_on_win)
        useGadgets.checked = boolFrom(state.bot.bot_uses_gadgets)
        stateCheck.text = String(state.time.state_check || 5)
        noDetect.text = String(state.time.no_detections || 10)
        idleField.text = String(state.time.idle || 5)
        gadgetField.text = String(state.time.gadget || 0.5)
        hyperField.text = String(state.time.hypercharge || 1.0)
        superField.text = String(state.time.super || 0.1)
        wallField.text = String(state.time.wall_detection || 0.2)
        noProceed.text = String(state.time.no_detection_proceed || 6.5)
        crashCheck.text = String(state.time.check_if_brawl_stars_crashed || 10)
        farmEnabled.checked = boolFrom(state.bot.smart_trophy_farm)
        farmTarget.text = String(state.bot.trophy_farm_target || 500)
        farmStrategy.currentIndex = Math.max(0, ["lowest_first","highest_first","in_order"].indexOf(String(state.bot.trophy_farm_strategy || "lowest_first")))
        questEnabled.checked = boolFrom(state.bot.quest_farm_enabled)
        questMode.currentIndex = Math.max(0, ["games","wins"].indexOf(String(state.bot.quest_farm_mode || "games")))
    }

    ListModel { id: excludeModel }
    ListModel { id: questExcludeModel }
    ListModel { id: gamemodeModel }
    ListModel { id: emulatorModel }

    function rebuildExcludeModels() {
        excludeModel.clear()
        questExcludeModel.clear()
        for (let i = 0; i < brawlers.length; ++i) {
            excludeModel.append({"name": brawlers[i].name, "displayName": brawlers[i].displayName, "checked": (state.bot.trophy_farm_excluded || []).indexOf(brawlers[i].name) !== -1})
            questExcludeModel.append({"name": brawlers[i].name, "displayName": brawlers[i].displayName, "checked": (state.bot.quest_farm_excluded || []).indexOf(brawlers[i].name) !== -1})
        }
    }

    component AppCard: Rectangle {
        radius: root.cardRadius
        color: root.panel
        border.color: root.border
        border.width: 1
    }

    component AppLabel: Label {
        color: root.textDim
        font.pixelSize: 12
    }

    component CardTitle: Label {
        color: root.textMain
        font.pixelSize: 20
        font.bold: true
    }

    component AppButton: Button {
        id: control
        implicitHeight: root.fieldHeight
        implicitWidth: 150
        font.pixelSize: 14
        background: Rectangle {
            radius: 14
            color: control.down ? "#31353D" : (control.hovered ? root.panelHover : "#2A2D32")
            border.color: control.highlighted ? root.accent : root.border
            border.width: control.highlighted ? 1.2 : 1
        }
        contentItem: Text {
            text: control.text
            color: root.textMain
            font: control.font
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
    }

    component AccentButton: Button {
        id: control
        implicitHeight: root.fieldHeight
        implicitWidth: 160
        font.pixelSize: 14
        font.bold: true
        background: Rectangle {
            radius: 14
            color: control.down ? "#B31E24" : (control.hovered ? "#F13A3A" : root.accent)
            border.color: "#FF6767"
            border.width: 1
        }
        contentItem: Text {
            text: control.text
            color: "#FFFFFF"
            font: control.font
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
    }

    component DestructiveButton: Button {
        id: control
        implicitHeight: root.fieldHeight
        implicitWidth: 140
        font.pixelSize: 14
        background: Rectangle {
            radius: 14
            color: control.down ? "#3B1417" : (control.hovered ? "#48181D" : "#301316")
            border.color: "#8C323C"
            border.width: 1
        }
        contentItem: Text {
            text: control.text
            color: root.textMain
            font: control.font
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
    }

    component NavButton: Button {
        id: control
        implicitHeight: 56
        font.pixelSize: 15
        background: Rectangle {
            radius: 16
            color: control.checked ? root.accentSoft : "transparent"
            border.color: control.checked ? root.accent : root.border
            border.width: 1
        }
        contentItem: Text {
            text: control.text
            color: control.checked ? root.textMain : root.textDim
            font.pixelSize: 15
            font.bold: control.checked
            leftPadding: 18
            verticalAlignment: Text.AlignVCenter
        }
    }

    component AppTextField: TextField {
        id: control
        implicitHeight: root.fieldHeight
        color: root.textMain
        placeholderTextColor: "#617089"
        selectedTextColor: "#FFFFFF"
        selectionColor: "#6D232A"
        font.pixelSize: 14
        padding: 14
        background: Rectangle {
            radius: 14
            color: root.fieldFill
            border.color: control.activeFocus ? root.accent : root.border
            border.width: control.activeFocus ? 1.2 : 1
        }
    }

    component AppComboBox: ComboBox {
        id: control
        implicitHeight: root.fieldHeight
        font.pixelSize: 14
        displayText: comboItemText(control.model, control.currentIndex, control.textRole)
        leftPadding: 14
        rightPadding: 42
        contentItem: Text {
            text: control.displayText
            color: root.textMain
            font: control.font
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
        background: Rectangle {
            radius: 14
            color: root.panelAlt
            border.color: control.activeFocus ? root.accent : root.border
            border.width: control.activeFocus ? 1.2 : 1
        }
        indicator: Label {
            text: "v"
            color: root.textDim
            font.pixelSize: 20
            anchors.verticalCenter: parent.verticalCenter
            anchors.right: parent.right
            anchors.rightMargin: 14
        }
        popup: Popup {
            y: control.height + 6
            width: control.width
            padding: 6
            background: Rectangle {
                radius: 14
                color: root.panelAlt
                border.color: root.border
                border.width: 1
            }
            contentItem: ListView {
                clip: true
                implicitHeight: Math.min(contentHeight, 260)
                model: control.popup.visible ? control.delegateModel : null
                delegate: control.delegate
                currentIndex: control.highlightedIndex
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { }
            }
        }
        delegate: ItemDelegate {
            width: control.width - 12
            implicitHeight: 40
            highlighted: control.highlightedIndex === index
            onClicked: {
                control.currentIndex = index
                control.activated(index)
                control.popup.close()
            }
            background: Rectangle {
                radius: 10
                color: highlighted ? root.accentSoft : "transparent"
            }
            contentItem: Text {
                text: comboItemText(control.model, index, control.textRole)
                color: root.textMain
                font.pixelSize: 14
                verticalAlignment: Text.AlignVCenter
                leftPadding: 10
                elide: Text.ElideRight
            }
        }
    }

    component AppCheckBox: CheckBox {
        id: control
        implicitHeight: 26
        font.pixelSize: 14
        spacing: 10
        indicator: Rectangle {
            implicitWidth: 24
            implicitHeight: 24
            radius: 6
            y: (control.height - height) / 2
            color: control.checked ? root.accent : "transparent"
            border.color: control.checked ? "#FF6D6D" : root.border
            border.width: 1
            Text {
                anchors.centerIn: parent
                text: control.checked ? "X" : ""
                color: "#FFFFFF"
                font.pixelSize: 15
                font.bold: true
            }
        }
        contentItem: Item {
            implicitWidth: labelText.implicitWidth + control.indicator.width + control.spacing
            implicitHeight: Math.max(control.indicator.height, labelText.implicitHeight)
            Text {
                id: labelText
                anchors.left: parent.left
                anchors.leftMargin: control.indicator.width + control.spacing
                anchors.verticalCenter: parent.verticalCenter
                text: control.text
                color: root.textMain
                font.pixelSize: 14
            }
        }
    }

    Component.onCompleted: {
        hydrate(backend.initialState())
        applyStateToForms()
        rebuildExcludeModels()
    }

    Connections {
        target: backend
        function onStateChanged(s) { hydrate(s); applyStateToForms(); rebuildExcludeModels() }
        function onRosterChanged(data) { roster = data; brawlers = backend.getBrawlers(); rebuildExcludeModels() }
        function onHistoryChanged(data) { history = data }
        function onLiveDataChanged(data) { live = data || {} }
        function onNotificationRaised(level, message) { notify(level, message) }
        function onSessionSummaryReady(summary) { summaryView.text = JSON.stringify(summary, null, 2); summaryPopup.open() }
    }

    Popup {
        id: toast
        x: root.width - width - 22
        y: 22
        width: Math.min(root.width * 0.38, 420)
        padding: 0
        background: Rectangle { radius: 14; color: Qt.darker(root.colorForLevel(root.toastLevel), 1.55); border.color: root.colorForLevel(root.toastLevel); border.width: 1 }
        contentItem: Text { text: root.toastText; color: root.textMain; font.pixelSize: 14; wrapMode: Text.WordWrap; padding: 14 }
        Timer { interval: 2600; running: toast.visible; onTriggered: toast.close() }
    }

    Popup {
        id: summaryPopup
        width: 720
        height: 520
        modal: true
        anchors.centerIn: parent
        padding: 0
        background: Rectangle { radius: 18; color: root.panel; border.color: root.border; border.width: 1 }
        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 18
            Label { text: "Session Summary"; color: root.textMain; font.pixelSize: 24; font.bold: true }
            TextArea { id: summaryView; Layout.fillWidth: true; Layout.fillHeight: true; readOnly: true; color: root.textMain; wrapMode: TextEdit.Wrap; background: Rectangle { radius: 12; color: root.panelAlt; border.color: root.border; border.width: 1 } }
            AppButton { text: "Close"; onClicked: summaryPopup.close() }
        }
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.preferredWidth: 258
            Layout.fillHeight: true
            color: root.sidebar
            border.color: root.border
            border.width: 1

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 18
                Label { text: "PYLAAI"; color: root.textMain; font.pixelSize: 30; font.bold: true; font.letterSpacing: 2 }
                Rectangle { width: 52; height: 5; radius: 3; color: root.accent }
                Label { text: state.branchLabel || ""; color: root.textDim; font.pixelSize: 13 }
                Repeater {
                    model: ["Control Center","Brawlers","Farm","Live","History","Settings"]
                    delegate: NavButton {
                        Layout.fillWidth: true
                        text: modelData
                        checkable: true
                        checked: pageIndex === index
                        onClicked: pageIndex = index
                    }
                }
                Item { Layout.fillHeight: true }
                AppCard {
                    Layout.fillWidth: true
                    implicitHeight: 94
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 6
                        Label { text: state.loggedIn ? "Connected" : "Offline / Local"; color: state.loggedIn ? root.success : root.warning; font.pixelSize: 14; font.bold: true }
                        Label { text: roster.length ? roster.length + " brawler(s) queued" : "No roster selected"; color: root.textDim; font.pixelSize: 13; wrapMode: Text.WordWrap }
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: root.bg
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: root.outerGap
                spacing: 18
                RowLayout {
                    Layout.fillWidth: true
                    Label { text: ["Control Center","Brawlers","Farm","Live Operations","Match History","Settings"][pageIndex]; color: root.textMain; font.pixelSize: 30; font.bold: true }
                    Item { Layout.fillWidth: true }
                    Rectangle {
                        radius: 999
                        color: (live.state || "").toLowerCase() === "match" ? "#12361F" : root.panel
                        border.color: (live.state || "").toLowerCase() === "match" ? root.success : root.border
                        border.width: 1
                        implicitWidth: liveBadge.implicitWidth + 28
                        implicitHeight: 40
                        Label { id: liveBadge; anchors.centerIn: parent; text: live.state ? String(live.state).toUpperCase() : "READY"; color: (live.state || "").toLowerCase() === "match" ? root.success : root.textDim; font.pixelSize: 14; font.bold: true }
                    }
                }

                StackLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    currentIndex: pageIndex

                    ScrollView {
                        id: controlCenterScroll
                        clip: true
                        contentWidth: availableWidth
                        Component.onCompleted: if (contentItem && contentItem.boundsBehavior !== undefined) contentItem.boundsBehavior = Flickable.StopAtBounds
                        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
                        Item {
                            width: controlCenterScroll.availableWidth
                            implicitHeight: controlCenterColumn.implicitHeight
                            Column {
                                id: controlCenterColumn
                                anchors.horizontalCenter: parent.horizontalCenter
                                width: Math.min(parent.width, 1320)
                                spacing: root.cardGap
                                AppCard {
                                    width: parent.width
                                    implicitHeight: 250
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 22
                                        spacing: 18
                                        CardTitle { text: "Launch Grid" }
                                        GridLayout {
                                            width: parent.width
                                            columns: 4
                                            columnSpacing: 14
                                            rowSpacing: 12
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Map Orientation" } AppComboBox { id: orientationBox; Layout.fillWidth: true; model: ["Vertical","Horizontal"] } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Gamemode" } AppComboBox { id: modeBox; Layout.fillWidth: true; model: gamemodeModel; textRole: "label" } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Emulator" } AppComboBox { id: emulatorBox; Layout.fillWidth: true; model: emulatorModel; textRole: "label" } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Run Minutes" } AppTextField { id: timerField; Layout.fillWidth: true } }
                                        }
                                        RowLayout {
                                            spacing: 12
                                            AppButton { text: "Save Controls"; onClicked: saveControl() }
                                            AccentButton { text: "Start Bot"; onClicked: backend.startBot() }
                                            DestructiveButton { text: "Stop Bot"; onClicked: backend.stopBot() }
                                        }
                                    }
                                }
                                AppCard {
                                    width: parent.width
                                    implicitHeight: Math.max(210, 70 + (roster.length * 96))
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 22
                                        spacing: 14
                                        CardTitle { text: "Selected Roster" }
                                        Label { visible: !roster.length; text: "No brawlers queued yet. Add them from the Brawlers page."; color: root.textDim; font.pixelSize: 14 }
                                        Repeater {
                                            model: roster
                                            delegate: Rectangle {
                                                width: parent.width
                                                height: 86
                                                radius: 16
                                                color: root.panelAlt
                                                border.color: root.border
                                                border.width: 1
                                                RowLayout {
                                                    anchors.fill: parent
                                                    anchors.margins: 14
                                                    spacing: 14
                                                    Rectangle {
                                                        Layout.preferredWidth: 56
                                                        Layout.preferredHeight: 56
                                                        radius: 14
                                                        color: "#0C0F14"
                                                        border.color: root.border
                                                        border.width: 1
                                                        clip: true
                                                        Image { anchors.fill: parent; source: modelData.icon || ""; fillMode: Image.PreserveAspectCrop; smooth: true; mipmap: true }
                                                    }
                                                    ColumnLayout {
                                                        Layout.fillWidth: true
                                                        spacing: 2
                                                        Label { text: modelData.displayName; color: root.textMain; font.pixelSize: 16; font.bold: true }
                                                        Label { text: modelData.type + " | target " + modelData.push_until; color: root.textDim; font.pixelSize: 12 }
                                                    }
                                                    Label { text: "Trophies " + modelData.trophies; color: root.gold; font.pixelSize: 14; font.bold: true }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Rectangle {
                        color: "transparent"
                        RowLayout {
                            anchors.fill: parent
                            spacing: root.cardGap
                            AppCard {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 20
                                    spacing: 14
                                    AppTextField { id: brawlerSearch; placeholderText: "Search brawlers"; Layout.fillWidth: true }
                                    ListView {
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true
                                        clip: true
                                        spacing: 10
                                        boundsBehavior: Flickable.StopAtBounds
                                        ScrollBar.vertical: ScrollBar { }
                                        model: brawlers.filter(function(item){ return !brawlerSearch.text || item.displayName.toLowerCase().indexOf(brawlerSearch.text.toLowerCase()) !== -1 })
                                        delegate: Rectangle {
                                            width: ListView.view.width
                                            height: 86
                                            radius: 16
                                            color: selectedBrawler === modelData.name ? root.accentSoft : root.panelAlt
                                            border.color: selectedBrawler === modelData.name ? root.accent : root.border
                                            border.width: 1
                                            RowLayout {
                                                anchors.fill: parent
                                                anchors.margins: 14
                                                spacing: 14
                                                Rectangle {
                                                    Layout.preferredWidth: 58
                                                    Layout.preferredHeight: 58
                                                    radius: 14
                                                    color: "#0C0F14"
                                                    border.color: root.border
                                                    border.width: 1
                                                    clip: true
                                                    Image { anchors.fill: parent; source: modelData.icon; fillMode: Image.PreserveAspectCrop; smooth: true; mipmap: true }
                                                }
                                                ColumnLayout {
                                                    Layout.fillWidth: true
                                                    spacing: 2
                                                    Label { text: modelData.displayName; color: root.textMain; font.pixelSize: 16; font.bold: true }
                                                    Label { text: "Hold attack " + modelData.holdAttack + "s"; color: root.textDim; font.pixelSize: 12 }
                                                }
                                                Label { text: modelData.trophies; color: root.gold; font.pixelSize: 14; font.bold: true }
                                            }
                                            MouseArea { anchors.fill: parent; onClicked: { selectedBrawler = modelData.name; hydrateEditors() } }
                                        }
                                    }
                                }
                            }
                            AppCard {
                                Layout.preferredWidth: 380
                                Layout.fillHeight: true
                                ColumnLayout {
                                    anchors.fill: parent
                                    anchors.margins: 20
                                    spacing: 14
                                    Label { id: brawlerTitle; text: "Select a brawler"; color: root.textMain; font.pixelSize: 24; font.bold: true }
                                    GridLayout {
                                        width: parent.width
                                        columns: 2
                                        columnSpacing: 12
                                        rowSpacing: 10
                                        ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Trophies" } AppTextField { id: trophiesField; Layout.fillWidth: true } }
                                        ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Wins" } AppTextField { id: winsField; Layout.fillWidth: true } }
                                        ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Push Until" } AppTextField { id: targetField; Layout.fillWidth: true } }
                                        ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Win Streak" } AppTextField { id: streakField; Layout.fillWidth: true } }
                                    }
                                    ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Push Type" } AppComboBox { id: typeBox; Layout.fillWidth: true; model: ["Trophies","Wins","Quest"] } }
                                    RowLayout {
                                        Layout.alignment: Qt.AlignHCenter
                                        spacing: 20
                                        AppCheckBox { id: autoPick; text: "Auto-pick" }
                                        AppCheckBox { id: manualTrophies; text: "Manual trophies" }
                                    }
                                        GridLayout {
                                            width: parent.width
                                            columns: 2
                                            columnSpacing: 12
                                            rowSpacing: 12
                                            AppButton { text: "Add / Update"; Layout.fillWidth: true; highlighted: true; onClicked: saveBrawler() }
                                            DestructiveButton { text: "Remove"; Layout.fillWidth: true; onClicked: backend.removeRosterEntry(selectedBrawler) }
                                            AppButton { text: "Load Config"; Layout.fillWidth: true; onClicked: backend.loadRosterFile() }
                                            AppButton { text: "Export"; Layout.fillWidth: true; onClicked: backend.exportRosterFile() }
                                            AppButton { text: "Import from Tag"; Layout.fillWidth: true; onClicked: backend.importSelectedBrawlerFromBrawlStarsApi(selectedBrawler) }
                                            Item { Layout.fillWidth: true }
                                        }
                                    DestructiveButton { text: "Clear Queue"; Layout.fillWidth: true; onClicked: backend.clearRoster() }
                                    CardTitle { text: "Current Queue" }
                                    ListView {
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true
                                        clip: true
                                        spacing: 10
                                        boundsBehavior: Flickable.StopAtBounds
                                        ScrollBar.vertical: ScrollBar { }
                                        model: roster
                                        delegate: Rectangle {
                                            width: ListView.view.width
                                            height: 74
                                            radius: 14
                                            color: root.panelAlt
                                            border.color: root.border
                                            border.width: 1
                                            RowLayout {
                                                anchors.fill: parent
                                                anchors.margins: 12
                                                spacing: 12
                                                Rectangle {
                                                    Layout.preferredWidth: 46
                                                    Layout.preferredHeight: 46
                                                    radius: 12
                                                    color: "#0C0F14"
                                                    border.color: root.border
                                                    border.width: 1
                                                    clip: true
                                                    Image { anchors.fill: parent; source: modelData.icon; fillMode: Image.PreserveAspectCrop; smooth: true; mipmap: true }
                                                }
                                                ColumnLayout { Layout.fillWidth: true; spacing: 2; Label { text: modelData.displayName; color: root.textMain; font.pixelSize: 15; font.bold: true } Label { text: modelData.type + " | target " + modelData.push_until; color: root.textDim; font.pixelSize: 12 } }
                                                Label { text: modelData.trophies; color: root.gold; font.pixelSize: 14; font.bold: true }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    ScrollView {
                        id: farmScroll
                        clip: true
                        contentWidth: availableWidth
                        Component.onCompleted: if (contentItem && contentItem.boundsBehavior !== undefined) contentItem.boundsBehavior = Flickable.StopAtBounds
                        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
                        Item {
                            width: farmScroll.availableWidth
                            implicitHeight: farmColumn.implicitHeight
                            Column {
                                id: farmColumn
                                anchors.horizontalCenter: parent.horizontalCenter
                                width: Math.min(parent.width, 1320)
                                spacing: root.cardGap
                                AppCard {
                                    width: parent.width
                                    implicitHeight: trophyFarmContent.implicitHeight + 44
                                    ColumnLayout {
                                        id: trophyFarmContent
                                        anchors.fill: parent
                                        anchors.margins: 22
                                        spacing: 16
                                        CardTitle { text: "Trophy Farm" }
                                        RowLayout {
                                            spacing: 12
                                            AppCheckBox { id: farmEnabled; text: "Enable" }
                                            AppTextField { id: farmTarget; implicitWidth: 140; placeholderText: "500" }
                                            AppComboBox { id: farmStrategy; implicitWidth: 190; model: ["lowest_first","highest_first","in_order"] }
                                            AppButton { text: "Save Farm"; onClicked: saveFarm() }
                                        }
                                        AppTextField {
                                            id: farmSearch
                                            Layout.fillWidth: true
                                            placeholderText: "Search excluded brawlers"
                                        }
                                        Label {
                                            text: checkedCount(excludeModel) + " excluded"
                                            color: root.textDim
                                            font.pixelSize: 13
                                        }
                                        Label { text: "Exclude brawlers from trophy farm"; color: root.textDim; font.pixelSize: 14 }
                                        ListView {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            Layout.preferredHeight: 220
                                            clip: true
                                            spacing: 8
                                            boundsBehavior: Flickable.StopAtBounds
                                            ScrollBar.vertical: ScrollBar { }
                                            id: trophyList
                                            model: excludeModel
                                            delegate: Rectangle {
                                                visible: !farmSearch.text || model.displayName.toLowerCase().indexOf(farmSearch.text.toLowerCase()) !== -1
                                                width: ListView.view.width
                                                height: visible ? 50 : 0
                                                radius: 12
                                                color: root.panelAlt
                                                border.color: root.border
                                                border.width: 1
                                                RowLayout {
                                                    anchors.fill: parent
                                                    anchors.margins: 12
                                                    spacing: 12
                                                    AppCheckBox { checked: model.checked; onToggled: excludeModel.setProperty(index, "checked", checked) }
                                                    Label { Layout.fillWidth: true; text: model.displayName; color: root.textMain; font.pixelSize: 14; horizontalAlignment: Text.AlignLeft; verticalAlignment: Text.AlignVCenter }
                                                }
                                            }
                                        }
                                    }
                                }
                                AppCard {
                                    width: parent.width
                                    implicitHeight: questFarmContent.implicitHeight + 44
                                    ColumnLayout {
                                        id: questFarmContent
                                        anchors.fill: parent
                                        anchors.margins: 22
                                        spacing: 16
                                        CardTitle { text: "Quest Farm" }
                                        Label { text: hasCapability("quest_farm") ? "Quest routing is available on this branch." : "Quest routing is not exposed on this branch."; color: root.textDim; font.pixelSize: 14; wrapMode: Text.WordWrap }
                                        RowLayout {
                                            visible: hasCapability("quest_farm")
                                            Layout.preferredHeight: visible ? implicitHeight : 0
                                            spacing: 12
                                            Layout.alignment: Qt.AlignVCenter
                                            AppCheckBox { id: questEnabled; text: "Enable" }
                                            AppComboBox { id: questMode; implicitWidth: 170; model: ["games","wins"] }
                                        }
                                        ListView {
                                            visible: hasCapability("quest_farm")
                                            Layout.fillWidth: true
                                            Layout.preferredHeight: visible ? 180 : 0
                                            clip: true
                                            spacing: 8
                                            boundsBehavior: Flickable.StopAtBounds
                                            ScrollBar.vertical: ScrollBar { }
                                            model: questExcludeModel
                                            delegate: Rectangle {
                                                width: ListView.view.width
                                                height: 50
                                                radius: 12
                                                color: root.panelAlt
                                                border.color: root.border
                                                border.width: 1
                                                RowLayout {
                                                    anchors.fill: parent
                                                    anchors.margins: 12
                                                    spacing: 12
                                                    AppCheckBox { checked: model.checked; onToggled: questExcludeModel.setProperty(index, "checked", checked) }
                                                    Label { Layout.fillWidth: true; text: model.displayName; color: root.textMain; font.pixelSize: 14; horizontalAlignment: Text.AlignLeft; verticalAlignment: Text.AlignVCenter }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    ScrollView {
                        id: liveScroll
                        clip: true
                        contentWidth: availableWidth
                        Component.onCompleted: if (contentItem && contentItem.boundsBehavior !== undefined) contentItem.boundsBehavior = Flickable.StopAtBounds
                        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
                        Item {
                            width: liveScroll.availableWidth
                            implicitHeight: liveColumn.implicitHeight
                            Column {
                                id: liveColumn
                                anchors.horizontalCenter: parent.horizontalCenter
                                width: Math.min(parent.width, 1320)
                                spacing: root.cardGap
                                GridLayout {
                                    width: parent.width
                                    columns: 3
                                    rowSpacing: root.cardGap
                                    columnSpacing: root.cardGap
                                    Repeater {
                                        model: [
                                            {"title":"Session","value": formatDuration(secondsSinceStart()), "color": root.info},
                                            {"title":"State","value": String(live.state || "READY").toUpperCase(), "color": (live.state || "").toLowerCase() === "match" ? root.success : root.textMain},
                                            {"title":"Brawler","value": live.brawler || "-", "color": root.textMain},
                                            {"title":"Trophies","value": liveMetricNumber(live.trophies, 0) + " / " + liveMetricNumber(live.target, 0), "color": root.gold},
                                            {"title":"To Target","value": String(Math.max(0, liveMetricNumber(live.target, 0) - liveMetricNumber(live.trophies, 0))), "color": root.warning},
                                            {"title":"IPS","value": Number(liveMetricNumber(live.ips, 0)).toFixed(1), "color": root.success},
                                            {"title":"KDA","value": live.current_deaths > 0 ? (liveMetricNumber(live.current_kills, 0) / Math.max(1, liveMetricNumber(live.current_deaths, 0))).toFixed(2) : String(liveMetricNumber(live.current_kills, 0)), "color": root.textMain},
                                            {"title":"DMG / MIN","value": String(secondsSinceStart() > 0 ? Math.round(liveMetricNumber(live.current_damage, 0) / Math.max(1, secondsSinceStart() / 60)) : 0), "color": root.textMain},
                                            {"title":"WINS / H","value": secondsSinceStart() > 0 ? ((liveMetricNumber(live.session_victories, 0) * 3600) / Math.max(1, secondsSinceStart())).toFixed(1) : "0.0", "color": root.textMain}
                                        ]
                                        delegate: AppCard {
                                            Layout.fillWidth: true
                                            implicitHeight: 118
                                            ColumnLayout {
                                                anchors.fill: parent
                                                anchors.margins: 18
                                                spacing: 10
                                                AppLabel { text: modelData.title.toUpperCase(); font.bold: true }
                                                Item { Layout.fillHeight: true }
                                                Label { text: String(modelData.value); color: modelData.color; font.pixelSize: 28; font.bold: true; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                                            }
                                        }
                                    }
                                }
                                AppCard {
                                    width: parent.width
                                    implicitHeight: 210
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: 22
                                        spacing: 14
                                        CardTitle { text: "Performance" }
                                        Label { text: "Session W / L / D: " + (live.session_victories || 0) + " / " + (live.session_defeats || 0) + " / " + (live.session_draws || 0); color: root.textMain; font.pixelSize: 15 }
                                        Label { text: "Current Match: " + (live.current_kills || 0) + " kills, " + (live.current_damage || 0) + " damage, " + (live.current_deaths || 0) + " deaths"; color: root.textDim; font.pixelSize: 14; wrapMode: Text.WordWrap }
                                        Label { text: "Last Match: " + (live.last_kills || 0) + " kills, " + (live.last_damage || 0) + " damage"; color: root.textDim; font.pixelSize: 14; wrapMode: Text.WordWrap }
                                        Label { visible: hasCapability("advanced_live"); text: "RL Episodes " + (live.rl_total_episodes || 0) + " | Buffer " + (live.rl_buffer_size || 0) + "/" + (live.rl_buffer_capacity || 0); color: root.gold; font.pixelSize: 14 }
                                    }
                                }
                            }
                        }
                    }

                    ScrollView {
                        id: historyScroll
                        clip: true
                        contentWidth: availableWidth
                        Component.onCompleted: if (contentItem && contentItem.boundsBehavior !== undefined) contentItem.boundsBehavior = Flickable.StopAtBounds
                        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
                        Item {
                            width: historyScroll.availableWidth
                            implicitHeight: historyColumn.implicitHeight
                            Column {
                                id: historyColumn
                                anchors.horizontalCenter: parent.horizontalCenter
                                width: Math.min(parent.width, 1320)
                                spacing: 12
                                Label { visible: !history.length; text: "No match history available yet."; color: root.textDim; font.pixelSize: 14 }
                                Repeater {
                                    model: history
                                    delegate: AppCard {
                                        width: historyColumn.width
                                        implicitHeight: 90
                                        RowLayout {
                                            anchors.fill: parent
                                            anchors.margins: 14
                                            spacing: 14
                                            Rectangle {
                                                Layout.preferredWidth: 58
                                                Layout.preferredHeight: 58
                                                radius: 14
                                                color: "#0C0F14"
                                                border.color: root.border
                                                border.width: 1
                                                clip: true
                                                Image { anchors.fill: parent; source: modelData.icon || ""; fillMode: Image.PreserveAspectCrop; smooth: true; mipmap: true }
                                            }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 2; Label { text: modelData.displayName; color: root.textMain; font.pixelSize: 17; font.bold: true } Label { text: modelData.wins + "W | " + modelData.defeats + "L | " + modelData.draws + "D"; color: root.textDim; font.pixelSize: 13 } }
                                            Label { text: modelData.matches + " matches"; color: root.info; font.pixelSize: 14; font.bold: true }
                                            Label { text: Number(modelData.winrate || 0).toFixed(1) + "%"; color: root.gold; font.pixelSize: 18; font.bold: true }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    ScrollView {
                        id: settingsScroll
                        clip: true
                        contentWidth: availableWidth
                        Component.onCompleted: if (contentItem && contentItem.boundsBehavior !== undefined) contentItem.boundsBehavior = Flickable.StopAtBounds
                        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
                        Item {
                            width: settingsScroll.availableWidth
                            implicitHeight: settingsColumn.implicitHeight
                            Column {
                                id: settingsColumn
                                anchors.horizontalCenter: parent.horizontalCenter
                                width: Math.min(parent.width, 1320)
                                spacing: root.cardGap
                                AppCard {
                                    width: parent.width
                                    implicitHeight: accountApiContent.implicitHeight + 44
                                    ColumnLayout {
                                        id: accountApiContent
                                        anchors.fill: parent
                                        anchors.margins: 22
                                        spacing: 16
                                        CardTitle { text: "Account & API" }
                                        GridLayout {
                                            width: parent.width
                                            columns: 2
                                            columnSpacing: 14
                                            rowSpacing: 12
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Pyla API Key" } AppTextField { id: pylaKey; Layout.fillWidth: true; echoMode: TextInput.PasswordEchoOnEdit } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Discord ID" } AppTextField { id: discordField; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Webhook" } AppTextField { id: webhookField; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "API Base URL" } AppTextField { id: apiBaseField; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Brawl Stars API Key" } AppTextField { id: bsApiField; Layout.fillWidth: true; echoMode: TextInput.PasswordEchoOnEdit } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Player Tag" } AppTextField { id: playerTagField; Layout.fillWidth: true } }
                                        }
                                    }
                                }
                                AppCard {
                                    width: parent.width
                                    implicitHeight: generalRuntimeContent.implicitHeight + 44
                                    ColumnLayout {
                                        id: generalRuntimeContent
                                        anchors.fill: parent
                                        anchors.margins: 22
                                        spacing: 16
                                        CardTitle { text: "General Runtime" }
                                        GridLayout {
                                            width: parent.width
                                            columns: 3
                                            columnSpacing: 14
                                            rowSpacing: 12
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Max IPS" } AppTextField { id: maxIps; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Backend" } AppComboBox { id: backendBox; Layout.fillWidth: true; model: ["auto","cpu","gpu"] } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Package" } AppTextField { id: packageField; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Run Minutes" } AppTextField { id: settingsTimer; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Port" } AppTextField { id: portField; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Auto Push" } AppTextField { id: autoPushField; Layout.fillWidth: true } }
                                        ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Emulator" } AppComboBox { id: settingsEmulator; Layout.fillWidth: true; model: emulatorModel; textRole: "label" } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Orientation" } AppComboBox { id: settingsOrientation; Layout.fillWidth: true; model: ["Vertical","Horizontal"] } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Debug" } Item { Layout.fillWidth: true; implicitHeight: root.fieldHeight; RowLayout { anchors.fill: parent; anchors.verticalCenter: parent.verticalCenter; AppCheckBox { id: debugBox; text: "Super debug" } } } }
                                        }
                                    }
                                }
                                AppCard {
                                    width: parent.width
                                    implicitHeight: combatDetectionContent.implicitHeight + 44
                                    ColumnLayout {
                                        id: combatDetectionContent
                                        anchors.fill: parent
                                        anchors.margins: 22
                                        spacing: 16
                                        CardTitle { text: "Combat & Detection" }
                                        GridLayout {
                                            width: parent.width
                                            columns: 3
                                            columnSpacing: 14
                                            rowSpacing: 12
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Min Move" } AppTextField { id: minMove; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Unstuck Delay" } AppTextField { id: unstuckDelay; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Unstuck Hold" } AppTextField { id: unstuckHold; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Wall Confidence" } AppTextField { id: wallConf; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Entity Confidence" } AppTextField { id: entityConf; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Hold Attack" } AppTextField { id: holdAttack; Layout.fillWidth: true } }
                                        }
                                        RowLayout { spacing: 18; AppCheckBox { id: playAgain; text: "Play again on win" } AppCheckBox { id: useGadgets; text: "Use gadgets" } }
                                        GridLayout {
                                            width: parent.width
                                            columns: 3
                                            columnSpacing: 14
                                            rowSpacing: 12
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "State Check" } AppTextField { id: stateCheck; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "No Detections" } AppTextField { id: noDetect; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Idle" } AppTextField { id: idleField; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Gadget" } AppTextField { id: gadgetField; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Hypercharge" } AppTextField { id: hyperField; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Super" } AppTextField { id: superField; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Wall Detect" } AppTextField { id: wallField; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "No Proceed" } AppTextField { id: noProceed; Layout.fillWidth: true } }
                                            ColumnLayout { Layout.fillWidth: true; spacing: 6; AppLabel { text: "Crash Check" } AppTextField { id: crashCheck; Layout.fillWidth: true } }
                                        }
                                        AccentButton { text: "Save Settings"; onClicked: saveSettings() }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}
