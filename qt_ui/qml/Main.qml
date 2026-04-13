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
    function hydrate(newState) {
        state = newState || {}
        roster = (state.roster || []).slice()
        brawlers = (state.brawlers || []).slice()
        history = (state.history || []).slice()
        live = state.live || {}
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
        backend.saveControlSettings({"map_orientation": orientationBox.currentText.toLowerCase(), "current_emulator": emulatorBox.currentText, "run_for_minutes": timerField.text, "gamemode": modeBox.currentValue})
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
        let emulators = state.emulators || []
        let emuIndex = emulators.indexOf(state.general.current_emulator || "LDPlayer")
        emulatorBox.currentIndex = Math.max(0, emuIndex)
        settingsEmulator.currentIndex = Math.max(0, emuIndex)
        let modes = state.gamemodes || []
        let gm = String(state.bot.gamemode || "knockout")
        let idx = 0
        for (let i = 0; i < modes.length; ++i) if (modes[i].value === gm) idx = i
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

    function rebuildExcludeModels() {
        excludeModel.clear()
        questExcludeModel.clear()
        for (let i = 0; i < brawlers.length; ++i) {
            excludeModel.append({"name": brawlers[i].name, "displayName": brawlers[i].displayName, "checked": (state.bot.trophy_farm_excluded || []).indexOf(brawlers[i].name) !== -1})
            questExcludeModel.append({"name": brawlers[i].name, "displayName": brawlers[i].displayName, "checked": (state.bot.quest_farm_excluded || []).indexOf(brawlers[i].name) !== -1})
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
            Button { text: "Close"; onClicked: summaryPopup.close() }
        }
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.preferredWidth: 238
            Layout.fillHeight: true
            color: root.sidebar
            border.color: root.border
            border.width: 1

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 18
                spacing: 16
                Label { text: "PYLAAI"; color: root.textMain; font.pixelSize: 28; font.bold: true; font.letterSpacing: 2 }
                Rectangle { width: 42; height: 4; radius: 2; color: root.accent }
                Label { text: state.branchLabel || ""; color: root.textDim; font.pixelSize: 13 }
                Repeater {
                    model: ["Control Center","Brawlers","Farm","Live","History","Settings"]
                    delegate: Button {
                        Layout.fillWidth: true
                        implicitHeight: 52
                        text: modelData
                        onClicked: pageIndex = index
                        background: Rectangle { radius: 14; color: pageIndex === index ? root.accentSoft : "transparent"; border.color: pageIndex === index ? root.accent : root.border; border.width: 1 }
                        contentItem: Text { text: parent.text; color: pageIndex === index ? root.textMain : root.textDim; font.pixelSize: 14; font.bold: pageIndex === index; verticalAlignment: Text.AlignVCenter; leftPadding: 18 }
                    }
                }
                Item { Layout.fillHeight: true }
                Rectangle {
                    Layout.fillWidth: true
                    radius: 14
                    color: root.panel
                    border.color: root.border
                    border.width: 1
                    implicitHeight: 88
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 14
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
                anchors.margins: 22
                spacing: 16
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
                        clip: true
                        contentWidth: availableWidth
                        ColumnLayout {
                            width: parent.width
                            spacing: 16
                            Rectangle {
                                Layout.fillWidth: true; radius: 18; color: root.panel; border.color: root.border; border.width: 1; implicitHeight: 210
                                ColumnLayout { anchors.fill: parent; anchors.margins: 18; spacing: 12
                                    Label { text: "Launch Grid"; color: root.textMain; font.pixelSize: 22; font.bold: true }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        ColumnLayout { Layout.fillWidth: true; Label { text: "Map Orientation"; color: root.textDim; font.pixelSize: 12 } ComboBox { id: orientationBox; Layout.fillWidth: true; model: ["Vertical","Horizontal"] } }
                                        ColumnLayout { Layout.fillWidth: true; Label { text: "Gamemode"; color: root.textDim; font.pixelSize: 12 } ComboBox { id: modeBox; Layout.fillWidth: true; model: state.gamemodes || []; textRole: "label"; property string currentValue: currentIndex >= 0 && currentIndex < (model ? model.length : 0) ? model[currentIndex].value : "knockout" } }
                                        ColumnLayout { Layout.fillWidth: true; Label { text: "Emulator"; color: root.textDim; font.pixelSize: 12 } ComboBox { id: emulatorBox; Layout.fillWidth: true; model: state.emulators || [] } }
                                        ColumnLayout { Layout.fillWidth: true; Label { text: "Run Minutes"; color: root.textDim; font.pixelSize: 12 } TextField { id: timerField; Layout.fillWidth: true } }
                                    }
                                    RowLayout {
                                        spacing: 12
                                        Button { text: "Save Controls"; onClicked: saveControl() }
                                        Button { text: "Start Bot"; onClicked: backend.startBot() }
                                        Button { text: "Stop Bot"; onClicked: backend.stopBot() }
                                    }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true; radius: 18; color: root.panel; border.color: root.border; border.width: 1; implicitHeight: 320
                                ColumnLayout { anchors.fill: parent; anchors.margins: 18; spacing: 12
                                    Label { text: "Selected Roster"; color: root.textMain; font.pixelSize: 22; font.bold: true }
                                    ListView {
                                        Layout.fillWidth: true; Layout.fillHeight: true; clip: true; spacing: 8; model: roster
                                        delegate: Rectangle {
                                            width: ListView.view.width; height: 70; radius: 14; color: root.panelAlt; border.color: root.border; border.width: 1
                                            RowLayout { anchors.fill: parent; anchors.margins: 12; spacing: 12
                                                Image { source: modelData.icon; width: 44; height: 44; fillMode: Image.PreserveAspectFit; smooth: true }
                                                ColumnLayout { Layout.fillWidth: true
                                                    Label { text: modelData.displayName; color: root.textMain; font.pixelSize: 15; font.bold: true }
                                                    Label { text: modelData.type + " | target " + modelData.push_until; color: root.textDim; font.pixelSize: 12 }
                                                }
                                                Label { text: "Trophies " + modelData.trophies; color: root.gold; font.pixelSize: 13; font.bold: true }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Rectangle {
                        color: "transparent"
                        RowLayout { anchors.fill: parent; spacing: 16
                            Rectangle {
                                Layout.fillWidth: true; Layout.fillHeight: true; radius: 18; color: root.panel; border.color: root.border; border.width: 1
                                ColumnLayout { anchors.fill: parent; anchors.margins: 18; spacing: 12
                                    TextField { id: brawlerSearch; placeholderText: "Search brawlers"; Layout.fillWidth: true }
                                    ListView {
                                        Layout.fillWidth: true; Layout.fillHeight: true; clip: true; spacing: 8
                                        model: brawlers.filter(function(item){ return !brawlerSearch.text || item.displayName.toLowerCase().indexOf(brawlerSearch.text.toLowerCase()) !== -1 })
                                        delegate: Rectangle {
                                            width: ListView.view.width; height: 72; radius: 14; color: selectedBrawler === modelData.name ? root.accentSoft : root.panelAlt; border.color: selectedBrawler === modelData.name ? root.accent : root.border; border.width: 1
                                            RowLayout { anchors.fill: parent; anchors.margins: 12; spacing: 12
                                                Image { source: modelData.icon; width: 48; height: 48; fillMode: Image.PreserveAspectFit; smooth: true }
                                                ColumnLayout { Layout.fillWidth: true
                                                    Label { text: modelData.displayName; color: root.textMain; font.pixelSize: 15; font.bold: true }
                                                    Label { text: "Hold attack " + modelData.holdAttack + "s"; color: root.textDim; font.pixelSize: 12 }
                                                }
                                                Label { text: modelData.trophies; color: root.gold; font.pixelSize: 13; font.bold: true }
                                            }
                                            MouseArea { anchors.fill: parent; onClicked: { selectedBrawler = modelData.name; hydrateEditors() } }
                                        }
                                    }
                                }
                            }
                            Rectangle {
                                Layout.preferredWidth: 420; Layout.fillHeight: true; radius: 18; color: root.panel; border.color: root.border; border.width: 1
                                ColumnLayout { anchors.fill: parent; anchors.margins: 18; spacing: 10
                                    Label { id: brawlerTitle; text: "Select a brawler"; color: root.textMain; font.pixelSize: 24; font.bold: true }
                                    RowLayout { Layout.fillWidth: true; ColumnLayout { Layout.fillWidth: true; Label { text: "Trophies"; color: root.textDim; font.pixelSize: 12 } TextField { id: trophiesField; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "Wins"; color: root.textDim; font.pixelSize: 12 } TextField { id: winsField; Layout.fillWidth: true } } }
                                    RowLayout { Layout.fillWidth: true; ColumnLayout { Layout.fillWidth: true; Label { text: "Push Until"; color: root.textDim; font.pixelSize: 12 } TextField { id: targetField; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "Win Streak"; color: root.textDim; font.pixelSize: 12 } TextField { id: streakField; Layout.fillWidth: true } } }
                                    ComboBox { id: typeBox; Layout.fillWidth: true; model: ["Trophies","Wins","Quest"] }
                                    RowLayout { CheckBox { id: autoPick; text: "Auto-pick" } CheckBox { id: manualTrophies; text: "Manual trophies" } }
                                    RowLayout { spacing: 12; Button { text: "Add / Update"; onClicked: saveBrawler() } Button { text: "Remove"; onClicked: backend.removeRosterEntry(selectedBrawler) } Button { text: "Clear"; onClicked: backend.clearRoster() } }
                                    RowLayout { spacing: 12; Button { text: "Load Config"; onClicked: backend.loadRosterFile() } Button { text: "Export"; onClicked: backend.exportRosterFile() } }
                                    Label { text: "Current Queue"; color: root.textMain; font.pixelSize: 18; font.bold: true }
                                    ListView {
                                        Layout.fillWidth: true; Layout.fillHeight: true; clip: true; spacing: 8; model: roster
                                        delegate: Rectangle {
                                            width: ListView.view.width; height: 60; radius: 12; color: root.panelAlt; border.color: root.border; border.width: 1
                                            RowLayout { anchors.fill: parent; anchors.margins: 10; spacing: 10
                                                Image { source: modelData.icon; width: 38; height: 38; fillMode: Image.PreserveAspectFit; smooth: true }
                                                ColumnLayout { Layout.fillWidth: true; Label { text: modelData.displayName; color: root.textMain; font.pixelSize: 14; font.bold: true } Label { text: modelData.type + " | " + modelData.push_until; color: root.textDim; font.pixelSize: 12 } }
                                                Label { text: modelData.trophies; color: root.gold; font.pixelSize: 13; font.bold: true }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    ScrollView {
                        clip: true
                        contentWidth: availableWidth
                        ColumnLayout { width: parent.width; spacing: 16
                            Rectangle {
                                Layout.fillWidth: true; radius: 18; color: root.panel; border.color: root.border; border.width: 1; implicitHeight: 250
                                ColumnLayout { anchors.fill: parent; anchors.margins: 18; spacing: 10
                                    Label { text: "Trophy Farm"; color: root.textMain; font.pixelSize: 22; font.bold: true }
                                    RowLayout { spacing: 12; CheckBox { id: farmEnabled; text: "Enable" } TextField { id: farmTarget; width: 120; placeholderText: "500" } ComboBox { id: farmStrategy; width: 170; model: ["lowest_first","highest_first","in_order"] } Button { text: "Save Farm"; onClicked: saveFarm() } }
                                    ListView {
                                        Layout.fillWidth: true; Layout.fillHeight: true; clip: true; spacing: 6; id: trophyList; model: excludeModel
                                        delegate: Rectangle {
                                            width: ListView.view.width; height: 42; radius: 10; color: root.panelAlt; border.color: root.border; border.width: 1
                                            RowLayout { anchors.fill: parent; anchors.leftMargin: 12; anchors.rightMargin: 12; CheckBox { checked: model.checked; onToggled: excludeModel.setProperty(index, "checked", checked) } Label { text: model.displayName; color: root.textMain; font.pixelSize: 13 } }
                                        }
                                    }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true; radius: 18; color: root.panel; border.color: root.border; border.width: 1; implicitHeight: state.capabilities && state.capabilities.quest_farm ? 240 : 120
                                ColumnLayout { anchors.fill: parent; anchors.margins: 18; spacing: 10
                                    Label { text: "Quest Farm"; color: root.textMain; font.pixelSize: 22; font.bold: true }
                                    Label { text: state.capabilities && state.capabilities.quest_farm ? "Quest routing is available on this branch." : "Quest routing is not exposed on this branch."; color: root.textDim; font.pixelSize: 14 }
                                    RowLayout { visible: state.capabilities && state.capabilities.quest_farm; spacing: 12; CheckBox { id: questEnabled; text: "Enable" } ComboBox { id: questMode; model: ["games","wins"] } }
                                    ListView {
                                        visible: state.capabilities && state.capabilities.quest_farm
                                        Layout.fillWidth: true; Layout.fillHeight: true; clip: true; spacing: 6; model: questExcludeModel
                                        delegate: Rectangle {
                                            width: ListView.view.width; height: 42; radius: 10; color: root.panelAlt; border.color: root.border; border.width: 1
                                            RowLayout { anchors.fill: parent; anchors.leftMargin: 12; anchors.rightMargin: 12; CheckBox { checked: model.checked; onToggled: questExcludeModel.setProperty(index, "checked", checked) } Label { text: model.displayName; color: root.textMain; font.pixelSize: 13 } }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    ScrollView {
                        clip: true
                        contentWidth: availableWidth
                        ColumnLayout { width: parent.width; spacing: 16
                            Repeater {
                                model: [
                                    {"title":"Session","value": live.start_time ? Math.max(0, Math.floor(Date.now()/1000 - live.start_time)) + "s" : "0s"},
                                    {"title":"IPS","value": live.ips ? Number(live.ips).toFixed(1) : "0.0"},
                                    {"title":"State","value": live.state || "starting"},
                                    {"title":"Brawler","value": live.brawler || "-"},
                                    {"title":"Trophies","value": (live.trophies || 0) + " / " + (live.target || 0)},
                                    {"title":"To Target","value": live.target ? Math.max(0, Number(live.target) - Number(live.trophies || 0)) : 0},
                                    {"title":"KDA","value": live.current_deaths > 0 ? ((live.current_kills || 0) / Math.max(1, live.current_deaths)).toFixed(2) : String(live.current_kills || 0)},
                                    {"title":"DMG / MIN","value": live.start_time ? Math.round((Number(live.current_damage || 0) / Math.max(1, ((Date.now()/1000 - live.start_time) / 60)))) : 0},
                                    {"title":"WINS / H","value": live.start_time ? (((Number(live.session_victories || 0) * 3600) / Math.max(1, (Date.now()/1000 - live.start_time)))).toFixed(1) : "0.0"}
                                ]
                                delegate: Rectangle {
                                    Layout.fillWidth: true; radius: 18; color: root.panel; border.color: root.border; border.width: 1; implicitHeight: 86
                                    RowLayout { anchors.fill: parent; anchors.margins: 16; Label { text: modelData.title; color: root.textDim; font.pixelSize: 12; Layout.preferredWidth: 150 } Label { text: String(modelData.value); color: root.textMain; font.pixelSize: 24; font.bold: true } }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true; radius: 18; color: root.panel; border.color: root.border; border.width: 1; implicitHeight: 170
                                ColumnLayout { anchors.fill: parent; anchors.margins: 18; spacing: 10
                                    Label { text: "Performance"; color: root.textMain; font.pixelSize: 22; font.bold: true }
                                    Label { text: "Session W / L / D: " + (live.session_victories || 0) + " / " + (live.session_defeats || 0) + " / " + (live.session_draws || 0); color: root.textMain; font.pixelSize: 15 }
                                    Label { text: "Current Match: " + (live.current_kills || 0) + " kills, " + (live.current_damage || 0) + " damage, " + (live.current_deaths || 0) + " deaths"; color: root.textDim; font.pixelSize: 14; wrapMode: Text.WordWrap }
                                    Label { text: "Last Match: " + (live.last_kills || 0) + " kills, " + (live.last_damage || 0) + " damage"; color: root.textDim; font.pixelSize: 14; wrapMode: Text.WordWrap }
                                    Label { visible: state.capabilities && state.capabilities.advanced_live; text: "RL Episodes " + (live.rl_total_episodes || 0) + " | Buffer " + (live.rl_buffer_size || 0) + "/" + (live.rl_buffer_capacity || 0); color: root.gold; font.pixelSize: 14 }
                                }
                            }
                        }
                    }

                    ListView {
                        clip: true; spacing: 10; model: history
                        delegate: Rectangle {
                            width: ListView.view.width; height: 86; radius: 16; color: root.panel; border.color: root.border; border.width: 1
                            RowLayout { anchors.fill: parent; anchors.margins: 14; spacing: 12
                                Image { source: modelData.icon; width: 54; height: 54; fillMode: Image.PreserveAspectFit; smooth: true }
                                ColumnLayout { Layout.fillWidth: true; Label { text: modelData.displayName; color: root.textMain; font.pixelSize: 18; font.bold: true } Label { text: modelData.wins + "W | " + modelData.defeats + "L | " + modelData.draws + "D"; color: root.textDim; font.pixelSize: 13 } }
                                Label { text: modelData.matches + " matches"; color: root.info; font.pixelSize: 14; font.bold: true }
                                Label { text: Number(modelData.winrate || 0).toFixed(1) + "%"; color: root.gold; font.pixelSize: 18; font.bold: true }
                            }
                        }
                    }

                    ScrollView {
                        clip: true
                        contentWidth: availableWidth
                        ColumnLayout { width: parent.width; spacing: 16
                            Rectangle {
                                Layout.fillWidth: true; radius: 18; color: root.panel; border.color: root.border; border.width: 1; implicitHeight: 260
                                ColumnLayout { anchors.fill: parent; anchors.margins: 18; spacing: 10
                                    Label { text: "Account & API"; color: root.textMain; font.pixelSize: 22; font.bold: true }
                                    RowLayout { Layout.fillWidth: true; ColumnLayout { Layout.fillWidth: true; Label { text: "Pyla API Key"; color: root.textDim; font.pixelSize: 12 } TextField { id: pylaKey; Layout.fillWidth: true; echoMode: TextInput.PasswordEchoOnEdit } } ColumnLayout { Layout.fillWidth: true; Label { text: "Discord ID"; color: root.textDim; font.pixelSize: 12 } TextField { id: discordField; Layout.fillWidth: true } } }
                                    RowLayout { Layout.fillWidth: true; ColumnLayout { Layout.fillWidth: true; Label { text: "Webhook"; color: root.textDim; font.pixelSize: 12 } TextField { id: webhookField; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "API Base URL"; color: root.textDim; font.pixelSize: 12 } TextField { id: apiBaseField; Layout.fillWidth: true } } }
                                    RowLayout { Layout.fillWidth: true; ColumnLayout { Layout.fillWidth: true; Label { text: "Brawl Stars API Key"; color: root.textDim; font.pixelSize: 12 } TextField { id: bsApiField; Layout.fillWidth: true; echoMode: TextInput.PasswordEchoOnEdit } } ColumnLayout { Layout.fillWidth: true; Label { text: "Player Tag"; color: root.textDim; font.pixelSize: 12 } TextField { id: playerTagField; Layout.fillWidth: true } } }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true; radius: 18; color: root.panel; border.color: root.border; border.width: 1; implicitHeight: 250
                                ColumnLayout { anchors.fill: parent; anchors.margins: 18; spacing: 10
                                    Label { text: "General Runtime"; color: root.textMain; font.pixelSize: 22; font.bold: true }
                                    RowLayout { Layout.fillWidth: true; ColumnLayout { Layout.fillWidth: true; Label { text: "Max IPS"; color: root.textDim; font.pixelSize: 12 } TextField { id: maxIps; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "Backend"; color: root.textDim; font.pixelSize: 12 } ComboBox { id: backendBox; Layout.fillWidth: true; model: ["auto","cpu","gpu"] } } ColumnLayout { Layout.fillWidth: true; Label { text: "Package"; color: root.textDim; font.pixelSize: 12 } TextField { id: packageField; Layout.fillWidth: true } } }
                                    RowLayout { Layout.fillWidth: true; ColumnLayout { Layout.fillWidth: true; Label { text: "Run Minutes"; color: root.textDim; font.pixelSize: 12 } TextField { id: settingsTimer; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "Port"; color: root.textDim; font.pixelSize: 12 } TextField { id: portField; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "Auto Push"; color: root.textDim; font.pixelSize: 12 } TextField { id: autoPushField; Layout.fillWidth: true } } }
                                    RowLayout { Layout.fillWidth: true; ColumnLayout { Layout.fillWidth: true; Label { text: "Emulator"; color: root.textDim; font.pixelSize: 12 } ComboBox { id: settingsEmulator; Layout.fillWidth: true; model: state.emulators || [] } } ColumnLayout { Layout.fillWidth: true; Label { text: "Orientation"; color: root.textDim; font.pixelSize: 12 } ComboBox { id: settingsOrientation; Layout.fillWidth: true; model: ["Vertical","Horizontal"] } } CheckBox { id: debugBox; text: "Super debug" } }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true; radius: 18; color: root.panel; border.color: root.border; border.width: 1; implicitHeight: 320
                                ColumnLayout { anchors.fill: parent; anchors.margins: 18; spacing: 10
                                    Label { text: "Combat & Detection"; color: root.textMain; font.pixelSize: 22; font.bold: true }
                                    RowLayout { Layout.fillWidth: true; ColumnLayout { Layout.fillWidth: true; Label { text: "Min Move"; color: root.textDim; font.pixelSize: 12 } TextField { id: minMove; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "Unstuck Delay"; color: root.textDim; font.pixelSize: 12 } TextField { id: unstuckDelay; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "Unstuck Hold"; color: root.textDim; font.pixelSize: 12 } TextField { id: unstuckHold; Layout.fillWidth: true } } }
                                    RowLayout { Layout.fillWidth: true; ColumnLayout { Layout.fillWidth: true; Label { text: "Wall Conf"; color: root.textDim; font.pixelSize: 12 } TextField { id: wallConf; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "Entity Conf"; color: root.textDim; font.pixelSize: 12 } TextField { id: entityConf; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "Hold Attack"; color: root.textDim; font.pixelSize: 12 } TextField { id: holdAttack; Layout.fillWidth: true } } }
                                    RowLayout { CheckBox { id: playAgain; text: "Play again on win" } CheckBox { id: useGadgets; text: "Use gadgets" } }
                                    RowLayout { Layout.fillWidth: true; ColumnLayout { Layout.fillWidth: true; Label { text: "State Check"; color: root.textDim; font.pixelSize: 12 } TextField { id: stateCheck; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "No Detections"; color: root.textDim; font.pixelSize: 12 } TextField { id: noDetect; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "Idle"; color: root.textDim; font.pixelSize: 12 } TextField { id: idleField; Layout.fillWidth: true } } }
                                    RowLayout { Layout.fillWidth: true; ColumnLayout { Layout.fillWidth: true; Label { text: "Gadget"; color: root.textDim; font.pixelSize: 12 } TextField { id: gadgetField; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "Hyper"; color: root.textDim; font.pixelSize: 12 } TextField { id: hyperField; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "Super"; color: root.textDim; font.pixelSize: 12 } TextField { id: superField; Layout.fillWidth: true } } }
                                    RowLayout { Layout.fillWidth: true; ColumnLayout { Layout.fillWidth: true; Label { text: "Wall Detect"; color: root.textDim; font.pixelSize: 12 } TextField { id: wallField; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "No Proceed"; color: root.textDim; font.pixelSize: 12 } TextField { id: noProceed; Layout.fillWidth: true } } ColumnLayout { Layout.fillWidth: true; Label { text: "Crash Check"; color: root.textDim; font.pixelSize: 12 } TextField { id: crashCheck; Layout.fillWidth: true } } }
                                    Button { text: "Save Settings"; onClicked: saveSettings() }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}
